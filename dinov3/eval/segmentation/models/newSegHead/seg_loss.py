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
        Compute Mask2Former loss with mask and dice loss implementation.
        """
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes + 1]
        pred_masks = outputs['pred_masks']    # [B, num_queries, H, W]
        
        batch_size, num_queries = pred_logits.shape[:2]
        device = pred_logits.device
        H, W = pred_masks.shape[-2:]
        
        # Ensure targets are valid
        num_model_classes = pred_logits.shape[-1]
        targets = torch.clamp(targets, 0, num_model_classes - 1)
        
        losses = {}
        total_class_loss = 0
        total_mask_loss = 0  
        total_dice_loss = 0
        
        for b in range(batch_size):
            gt_mask = targets[b]  # [H, W]
            
            # Get unique classes in ground truth (excluding background if 0)
            unique_classes = torch.unique(gt_mask)
            if len(unique_classes) > 1 and unique_classes[0] == 0:
                unique_classes = unique_classes[1:]  # Remove background
            
            # Initialize classification and mask targets
            class_targets = torch.full(
                (num_queries,), num_model_classes - 1, dtype=torch.long, device=device
            )  # Background class
            mask_targets = torch.zeros((num_queries, H, W), device=device)
            
            # Assign ground truth segments to queries
            num_gt_segments = min(len(unique_classes), num_queries)
            
            for i, class_id in enumerate(unique_classes[:num_gt_segments]):
                if i < num_queries:
                    # Assign class
                    class_targets[i] = class_id.long()
                    
                    # Create binary mask for this class
                    binary_mask = (gt_mask == class_id).float()
                    mask_targets[i] = binary_mask
            
            # Classification loss
            try:
                class_loss = F.cross_entropy(pred_logits[b], class_targets)
                total_class_loss += class_loss
            except Exception as e:
                print(f"ERROR in cross_entropy: {e}")
                total_class_loss += torch.tensor(0.0, device=device, requires_grad=True)
            
            # Mask losses (BCE + Dice)
            pred_masks_b = pred_masks[b]  # [num_queries, H, W]
            
            # Only compute mask losses for non-background queries
            valid_mask_indices = (class_targets != num_model_classes - 1).nonzero(as_tuple=True)[0]
            
            if len(valid_mask_indices) > 0:
                # BCE Loss for masks
                pred_masks_valid = pred_masks_b[valid_mask_indices]  # [N, H, W]
                mask_targets_valid = mask_targets[valid_mask_indices]  # [N, H, W]
                
                # Apply sigmoid to predictions for BCE loss
                mask_bce_loss = self.mask_bce_loss(pred_masks_valid, mask_targets_valid)
                total_mask_loss += mask_bce_loss
                
                # Dice Loss for masks
                # Apply sigmoid to get probabilities
                pred_masks_prob = torch.sigmoid(pred_masks_valid)
                
                # Compute dice loss for each valid mask
                dice_loss = 0
                for i in range(len(valid_mask_indices)):
                    pred_flat = pred_masks_prob[i].view(-1)
                    target_flat = mask_targets_valid[i].view(-1)
                    
                    intersection = (pred_flat * target_flat).sum()
                    union = pred_flat.sum() + target_flat.sum()
                    
                    dice_coeff = (2.0 * intersection + self.dice_loss.smooth) / (union + self.dice_loss.smooth)
                    dice_loss += (1.0 - dice_coeff)
                
                if len(valid_mask_indices) > 0:
                    dice_loss = dice_loss / len(valid_mask_indices)
                
                total_dice_loss += dice_loss
        
        # Average losses across batch
        losses['class_loss'] = total_class_loss / batch_size
        losses['mask_loss'] = total_mask_loss / batch_size if total_mask_loss > 0 else torch.tensor(0.0, device=device)
        losses['dice_loss'] = total_dice_loss / batch_size if total_dice_loss > 0 else torch.tensor(0.0, device=device)
        
        # Total weighted loss
        total_loss = (self.weight_class * losses['class_loss'] + 
                     self.weight_mask * losses['mask_loss'] + 
                     self.weight_dice * losses['dice_loss'])
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