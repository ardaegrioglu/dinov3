import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt


class ImageNetSDataset(Dataset):
    """
    ImageNet-S dataset loader for segmentation training.
    Provides both images and ground truth segmentation masks.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        resolution: int = 224,
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root_dir: Path to ImageNet-S dataset root
            split: 'train' or 'val'
            resolution: Input image resolution (224)
            transform: Image transforms
            mask_transform: Mask transforms
        """
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution
        
        # Dataset structure:
        # root_dir/
        #   ├── ImageNetS50/
        #   │   ├── train-semi/  (or train-full)
        #   │   └── validation/
        #   └── ImageNetS300/
        #       ├── train-semi/
        #       └── validation/
        
        # Use ImageNetS919 (full dataset) for maximum classes
        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'ImageNetS919', 'train-semi')
            self.mask_dir = os.path.join(root_dir, 'ImageNetS919', 'train-semi')
        else:
            self.image_dir = os.path.join(root_dir, 'ImageNetS919', 'validation')
            self.mask_dir = os.path.join(root_dir, 'ImageNetS919', 'validation')
        
        # Load class information
        self.class_to_idx = self._load_class_mapping()
        self.num_classes = len(self.class_to_idx)
        
        # Get all image paths and corresponding mask paths
        self.samples = self._make_dataset()
        
        # Default transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
        if mask_transform is None:
            self.mask_transform = self._get_default_mask_transform()
        else:
            self.mask_transform = mask_transform
    
    def _load_class_mapping(self) -> Dict[str, int]:
        """Load class name to index mapping."""
        class_dirs = [d for d in os.listdir(self.image_dir) 
                     if os.path.isdir(os.path.join(self.image_dir, d))]
        class_dirs.sort()
        
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
        print(f"Found {len(class_to_idx)} classes in {self.split} split")
        
        return class_to_idx
    
    def _make_dataset(self) -> List[Tuple[str, str, int]]:
        """Create list of (image_path, mask_path, class_index) tuples."""
        samples = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_image_dir = os.path.join(self.image_dir, class_name)
            class_mask_dir = os.path.join(self.mask_dir, class_name)
            
            if not os.path.isdir(class_image_dir):
                continue
                
            # Get all images in this class
            for img_name in os.listdir(class_image_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(class_image_dir, img_name)
                
                # Corresponding mask path (assuming same name with .png extension)
                mask_name = os.path.splitext(img_name)[0] + '.png'
                mask_path = os.path.join(class_mask_dir, mask_name)
                
                # Only include samples that have both image and mask
                if os.path.exists(mask_path):
                    samples.append((img_path, mask_path, class_idx))
        
        print(f"Found {len(samples)} samples in {self.split} split")
        return samples
    
    def _get_default_transform(self) -> transforms.Compose:
        """Default image transforms for training/validation."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(self.resolution, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ])
    
    def _get_default_mask_transform(self) -> transforms.Compose:
        """Default mask transforms."""
        return transforms.Compose([
            transforms.Resize((self.resolution, self.resolution), 
                            interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.
        
        Returns:
            Dictionary containing:
                - image: [3, H, W] normalized image tensor
                - mask: [H, W] segmentation mask tensor with class indices
                - class_id: scalar class index
        """
        img_path, mask_path, class_idx = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply transforms
        if self.split == 'train':
            # For training, apply same random transform to both image and mask
            seed = np.random.randint(2147483647)
            
            # Transform image
            np.random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            
            # Transform mask with same random seed
            np.random.seed(seed) 
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        else:
            image = self.transform(image)
            mask = self.mask_transform(mask)
        
        # Convert mask to long tensor and squeeze
        mask = (mask * 255).long().squeeze(0)  # Remove channel dimension
        
        # Ensure mask values are in valid range [0, num_classes]
        mask = torch.clamp(mask, 0, self.num_classes)
        
        return {
            'image': image,
            'mask': mask,
            'class_id': torch.tensor(class_idx, dtype=torch.long),
            'image_path': img_path,
        }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return list(self.class_to_idx.keys())
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a sample from the dataset."""
        sample = self[idx]
        
        # Denormalize image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = sample['image'] * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot image
        axes[0].imshow(image.permute(1, 2, 0))
        axes[0].set_title(f"Image (Class: {sample['class_id'].item()})")
        axes[0].axis('off')
        
        # Plot mask
        mask = sample['mask'].numpy()
        axes[1].imshow(mask, cmap='tab20')
        axes[1].set_title(f"Segmentation Mask")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def create_imagenet_s_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    resolution: int = 224,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for ImageNet-S.
    
    Args:
        root_dir: Path to ImageNet-S dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        resolution: Input resolution
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = ImageNetSDataset(
        root_dir=root_dir,
        split='train',
        resolution=resolution,
    )
    
    val_dataset = ImageNetSDataset(
        root_dir=root_dir,
        split='val',
        resolution=resolution,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader

'''

## Key Features of this Dataset Loader:

1. **ImageNet-S Support**: Handles the ImageNetS919 dataset structure with both images and segmentation masks

2. **Expected Dataset Structure**:
   root_dir/
   ├── ImageNetS919/
   │   ├── train-semi/
   │   │   ├── class1/
   │   │   │   ├── image1.jpg
   │   │   │   ├── image1.png (mask)
   │   │   │   └── ...
   │   │   └── ...
   │   └── validation/
   │       └── ...
'''