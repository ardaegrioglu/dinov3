import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Computes the Dice coefficient between predicted and target masks.
    """
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted masks [B, C, H, W] (logits)
            target: Target masks [B, H, W] (class indices)
            
        Returns:
            Dice loss scalar
        """
        # Apply softmax to get probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten for computation
        pred_flat = pred_probs.view(pred_probs.shape[0], pred_probs.shape[1], -1)
        target_flat = target_one_hot.view(target_one_hot.shape[0], target_one_hot.shape[1], -1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)
        pred_sum = pred_flat.sum(dim=2)
        target_sum = target_flat.sum(dim=2)
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Return 1 - dice as loss
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            pred: Predicted logits [B, C, H, W]
            target: Target class indices [B, H, W]
            
        Returns:
            Focal loss scalar
        """
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class Mask2FormerLoss(nn.Module):
    """
    Loss function specifically for Mask2Former-style models.
    Handles the matching between predicted queries and ground truth masks.
    """
    
    def __init__(
        self,
        num_classes: int,
        weight_class: float = 2.0,
        weight_mask: float = 5.0,
        weight_dice: float = 5.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.weight_class = weight_class
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        
        # Classification loss (focal loss)
        self.class_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # Mask BCE loss
        self.mask_bce_loss = nn.BCEWithLogitsLoss()
        
        # Dice loss for masks
        self.dice_loss = DiceLoss(smooth=1.0)
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Mask2Former loss.
        
        Args:
            outputs: Model outputs containing 'pred_logits' and 'pred_masks'
            targets: Ground truth masks [B, H, W]
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes + 1]
        pred_masks = outputs['pred_masks']    # [B, num_queries, H, W]
        
        batch_size, num_queries = pred_logits.shape[:2]
        device = pred_logits.device
        
        # Create targets for each query (simplified matching)
        # In practice, you'd use Hungarian matching, but for simplicity:
        # We'll create targets by finding the best matching query for each ground truth
        
        losses = {}
        total_class_loss = 0
        total_mask_loss = 0
        total_dice_loss = 0
        
        for b in range(batch_size):
            gt_mask = targets[b]  # [H, W]
            
            # Create dummy classification targets (background for most queries)
            class_targets = torch.full(
                (num_queries,), self.num_classes, dtype=torch.long, device=device
            )  # Background class
            
            # For simplicity, assume first query should predict the mask
            if gt_mask.max() > 0:  # If there's a foreground object
                class_targets[0] = gt_mask.max().long()  # Use the max class as target
                
                # Create binary mask for the first query
                binary_gt = (gt_mask == gt_mask.max()).float()
                mask_target = binary_gt.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                mask_pred = pred_masks[b:b+1, 0:1]  # [1, 1, H, W]
                
                # Mask BCE loss
                mask_loss = self.mask_bce_loss(mask_pred, mask_target)
                total_mask_loss += mask_loss
                
                # Dice loss for mask
                dice_loss = self.dice_loss(mask_pred, binary_gt.unsqueeze(0).long())
                total_dice_loss += dice_loss
            
            # Classification loss
            class_loss = F.cross_entropy(pred_logits[b], class_targets)
            total_class_loss += class_loss
        
        # Average losses
        losses['class_loss'] = total_class_loss / batch_size
        losses['mask_loss'] = total_mask_loss / batch_size if total_mask_loss > 0 else torch.tensor(0.0, device=device)
        losses['dice_loss'] = total_dice_loss / batch_size if total_dice_loss > 0 else torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = (
            self.weight_class * losses['class_loss'] +
            self.weight_mask * losses['mask_loss'] +
            self.weight_dice * losses['dice_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses


def create_loss_function(
    num_classes: int = 919,
    **kwargs
) -> nn.Module:
    """
    Factory function to create Mask2Former loss function.
    
    Args:
        num_classes: Number of classes
        **kwargs: Additional arguments for loss function
        
    Returns:
        Mask2FormerLoss module
    """
    return Mask2FormerLoss(num_classes=num_classes, **kwargs)
