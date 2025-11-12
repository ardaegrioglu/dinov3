
"""
import os
import gc
import argparse
import math
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Import DINOv3 utilities
from dinov3.logging import MetricLogger, SmoothedValue, setup_logging
from dinov3.checkpointer import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    keep_last_n_checkpoints,
)
from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
from dinov3.data import SamplerType, make_data_loader
import dinov3.distributed as distributed

# Import Mask2Former head
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead

# Import custom modules
from imagenet_s_dataset import ImageNetSDataset
from seg_loss import create_loss_function

logger = logging.getLogger("dinov3")


class DINOv3Mask2FormerModel(nn.Module):
    
    # Simple wrapper combining DINOv3 backbone with Mask2Former head.
    
    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        num_classes: int = 919,
        pretrained_weights: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Load DINOv3 backbone
        if checkpoint_path:
            # Load from your trained checkpoint
            logger.info(f"Loading backbone from checkpoint: {checkpoint_path}")
            import dinov3.models.vision_transformer as vit
            
            # Parse backbone architecture from name
            if 'vits14' in backbone_name or 'vit_small' in backbone_name:
                self.backbone = vit.vit_small(patch_size=14)
            elif 'vitb14' in backbone_name or 'vit_base' in backbone_name:
                self.backbone = vit.vit_base(patch_size=14)
            elif 'vitl14' in backbone_name or 'vitl16' in backbone_name or 'vit_large' in backbone_name or 'vitl' in backbone_name:
                # Based on your config: vit_large with patch_size 16
                self.backbone = vit.vit_large(patch_size=16)
            elif 'vitg14' in backbone_name or 'vit_giant' in backbone_name:
                self.backbone = vit.vit_giant2(patch_size=14)
            else:
                raise ValueError(f"Unknown backbone: {backbone_name}")
            
            # Load checkpoint
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract student/backbone weights from checkpoint
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'student' in checkpoint:
                state_dict = checkpoint['student']
            elif 'teacher' in checkpoint:
                # If only teacher is available, use that
                logger.info("Using teacher weights from checkpoint")
                state_dict = checkpoint['teacher']
            else:
                state_dict = checkpoint
            
            # Filter to only backbone/student weights (remove teacher, dino head, ibot head, etc.)
            backbone_state_dict = {}
            for key, value in state_dict.items():
                # Remove 'student.', 'backbone.', or 'module.' prefixes
                new_key = key
                for prefix in ['student.', 'backbone.', 'module.', '_orig_mod.']:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                
                # Skip teacher, head, and other non-backbone weights
                if any(skip in key for skip in ['teacher', 'dino_head', 'ibot_head', 'head']):
                    continue
                
                backbone_state_dict[new_key] = value
            
            logger.info(f"Extracted {len(backbone_state_dict)} keys for backbone")
            
            # Load weights into backbone
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading backbone ({len(missing_keys)} keys): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading backbone ({len(unexpected_keys)} keys): {unexpected_keys[:5]}...")
            
            logger.info(f"Successfully loaded backbone from checkpoint")
            
        else:
            # Original loading logic (torch.hub or local)
            try:
                self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
                logger.info(f"Loaded {backbone_name} from torch.hub")
            except Exception as e:
                logger.warning(f"Failed to load from torch.hub: {e}")
                import dinov3.models.vision_transformer as vit
                if 'vits14' in backbone_name:
                    self.backbone = vit.vit_small(patch_size=14)
                elif 'vitb14' in backbone_name:
                    self.backbone = vit.vit_base(patch_size=14)
                elif 'vitl14' in backbone_name:
                    self.backbone = vit.vit_large(patch_size=14)
                elif 'vitg14' in backbone_name:
                    self.backbone = vit.vit_giant2(patch_size=14)
                else:
                    raise ValueError(f"Unknown backbone: {backbone_name}")
                logger.info(f"Loaded {backbone_name} from local module")
        
        # Get backbone dimension
        if 'vits14' in backbone_name or 'vit_small' in backbone_name:
            self.backbone_dim = 384
        elif 'vitb14' in backbone_name or 'vit_base' in backbone_name:
            self.backbone_dim = 768
        elif 'vitl14' in backbone_name or 'vitl16' in backbone_name or 'vit_large' in backbone_name or 'vitl' in backbone_name:
            self.backbone_dim = 1024
        elif 'vitg14' in backbone_name or 'vit_giant' in backbone_name:
            self.backbone_dim = 1536
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")
        
        # Feature projection and pyramid for multi-scale features
        self.feature_proj = nn.Conv2d(self.backbone_dim, hidden_dim, kernel_size=1)
        
        # Create multi-scale feature pyramid
        self.pyramid_layers = nn.ModuleDict({
            'scale1': nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=4),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(),
            ),
            'scale2': nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(),
            ),
            'scale3': nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(),
            ),
            'scale4': nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(),
            ),
        })
        
        # Define input shape for Mask2Former head
        # Format: {feature_name: (channels, height, width, stride)}
        input_shape = {
            "1": (hidden_dim, 56, 56, 4),
            "2": (hidden_dim, 28, 28, 8),
            "3": (hidden_dim, 14, 14, 16),
            "4": (hidden_dim, 7, 7, 32),
        }
        
        # Create Mask2Former head using the provided implementation
        self.seg_head = Mask2FormerHead(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            loss_weight=1.0,
            ignore_value=255,
        )
        
        logger.info(f"Model created: {backbone_name}, {num_classes} classes, hidden_dim={hidden_dim}")
    
    def extract_features(self, x):
        #Extract features from DINOv3 backbone.
        # Forward through backbone
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        # Handle different output formats
        if isinstance(features, dict):
            if 'x_norm_patchtokens' in features:
                features = features['x_norm_patchtokens']
            elif 'patch_tokens' in features:
                features = features['patch_tokens']
            else:
                raise ValueError(f"Unknown feature dict keys: {features.keys()}")
        
        # Reshape from [B, N, D] to [B, D, H, W]
        if features.dim() == 3:
            B, N, D = features.shape
            h = w = int(N ** 0.5)
            features = features.transpose(1, 2).reshape(B, D, h, w)
        
        return features
    
    def create_pyramid(self, features):
        #Create multi-scale feature pyramid.
        # Project to hidden dimension
        x = self.feature_proj(features)
        
        # Create multi-scale features
        pyramid = {
            '1': self.pyramid_layers['scale1'](x),
            '2': self.pyramid_layers['scale2'](x),
            '3': self.pyramid_layers['scale3'](x),
            '4': self.pyramid_layers['scale4'](x),
        }
        return pyramid
    
    def forward(self, x):
        #Forward pass.
        # Extract backbone features
        features = self.extract_features(x)
        
        # Create multi-scale pyramid
        pyramid = self.create_pyramid(features)
        
        # Forward through Mask2Former head
        outputs = self.seg_head(pyramid)
        
        return outputs


def build_schedulers(cfg: Dict[str, Any], iter_per_epoch: int):
    #Build learning rate and weight decay schedules as numpy arrays.
    total_iterations = cfg['epochs'] * iter_per_epoch
    logger.info(f"Total training iterations: {total_iterations}")
    
    # Learning rate schedule with warmup and cosine decay
    lr_peak = cfg['lr']
    lr_end = cfg['lr'] * 0.01  # End at 1% of peak
    
    lr_schedule = linear_warmup_cosine_decay(
        start=0.0,  # Start from 0 for warmup
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=iter_per_epoch * cfg.get('warmup_epochs', 5),
        total_iterations=total_iterations,
    )
    
    # Weight decay schedule
    wd_schedule = linear_warmup_cosine_decay(
        start=cfg['weight_decay'],
        peak=cfg['weight_decay'],
        end=cfg['weight_decay'],
        warmup_iterations=0,
        total_iterations=total_iterations,
    )
    
    # Separate learning rate for backbone (lower)
    backbone_lr_schedule = lr_schedule * cfg.get('backbone_lr_multiplier', 0.1)
    
    logger.info("Schedulers ready.")
    return lr_schedule, wd_schedule, backbone_lr_schedule


def apply_optim_scheduler(optimizer, lr, wd, backbone_lr):
    #Apply learning rate and weight decay schedules to optimizer.
    for param_group in optimizer.param_groups:
        is_backbone = param_group['name'] == 'backbone'
        param_group['weight_decay'] = wd
        if is_backbone:
            param_group['lr'] = backbone_lr
        else:
            param_group['lr'] = lr


class Mask2FormerSegmentationTrainer:
    #Trainer class for DINOv3 + Mask2Former segmentation on ImageNet-S.
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directory structure
        self.output_dir = Path(config['output_dir']).expanduser()
        self.ckpt_dir = self.output_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {config_path}")
        
        # Initialize WandB
        self.use_wandb = config.get('use_wandb', False)
        self.wandb_log_freq = config.get('wandb_log_freq', 10)
        if self.use_wandb and distributed.is_main_process():
            import wandb
            self.wandb_run = wandb.init(
                project=config.get('wandb_project', 'dinov3-mask2former-segmentation'),
                name=config.get('wandb_run_name', None),
                config=config,
                resume='allow',
                id=config.get('wandb_run_id', None),
            )
            logger.info(f"WandB initialized: {self.wandb_run.name}")
        else:
            self.wandb_run = None
        
        # Initialize model
        self.model = self._create_model()
        
        # Watch model with WandB
        if self.wandb_run is not None:
            self.wandb_run.watch(
                self.model, 
                log='all',
                log_freq=config.get('wandb_watch_freq', 1000),
            )
        
        # Initialize dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Calculate iterations per epoch and total iterations
        self.iter_per_epoch = len(self.train_loader)
        self.max_iter = config['epochs'] * self.iter_per_epoch
        logger.info(f"Iterations per epoch: {self.iter_per_epoch}")
        logger.info(f"Total iterations: {self.max_iter}")
        
        # Build schedulers
        self.lr_schedule, self.wd_schedule, self.backbone_lr_schedule = build_schedulers(
            config, self.iter_per_epoch
        )
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Training state
        self.start_iter = 0
        self.best_val_loss = float('inf')
        
        # Checkpointing parameters
        self.checkpoint_period = config.get('checkpoint_period', self.iter_per_epoch)
        self.eval_period = config.get('eval_period_iterations', self.iter_per_epoch)
        self.max_checkpoints_to_keep = config.get('max_checkpoints_to_keep', 3)
    
    def _create_model(self) -> nn.Module:
        # Create and initialize the Mask2Former segmentation model.
        model = DINOv3Mask2FormerModel(
            backbone_name=self.config['backbone_name'],
            num_classes=self.config['num_classes'],
            pretrained_weights=self.config.get('pretrained_weights', None),
            checkpoint_path=self.config.get('checkpoint_path', None),
            freeze_backbone=self.config.get('freeze_backbone', False),
            hidden_dim=self.config.get('hidden_dim', 256),
        )
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _create_dataloaders(self) -> tuple:
        #Create train and validation dataloaders using DINOv3 patterns.
        # Create datasets
        train_dataset = ImageNetSDataset(
            root_dir=self.config['dataset_path'],
            split='train',
            resolution=self.config['input_resolution'],
        )
        
        val_dataset = ImageNetSDataset(
            root_dir=self.config['dataset_path'],
            split='val',
            resolution=self.config['input_resolution'],
        )
        
        # Create dataloaders with DINOv3's make_data_loader
        train_loader = make_data_loader(
            dataset=train_dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=True,
            seed=self.config.get('seed', 0),
            sampler_type=SamplerType.INFINITE,
            sampler_advance=self.start_iter,
            drop_last=True,
            persistent_workers=True,
        )
        
        val_loader = make_data_loader(
            dataset=val_dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            seed=self.config.get('seed', 0),
            sampler_type=SamplerType.DISTRIBUTED,
            drop_last=False,
            persistent_workers=False,
        )
        
        logger.info(f"Train batches per epoch: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def _create_loss_function(self) -> nn.Module:
        #Create the Mask2Former loss function.
        return create_loss_function(
            loss_type='mask2former',
            num_classes=self.config['num_classes'],
            weight_class=self.config.get('weight_class', 2.0),
            weight_mask=self.config.get('weight_mask', 5.0),
            weight_dice=self.config.get('weight_dice', 5.0),
            focal_alpha=self.config.get('focal_alpha', 0.25),
            focal_gamma=self.config.get('focal_gamma', 2.0),
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        #Create optimizer with separate parameter groups for backbone and head.
        # Separate parameters for backbone and head
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        
        # Create parameter groups with names for scheduler application
        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.config['lr'] * self.config.get('backbone_lr_multiplier', 0.1),
                'name': 'backbone',
                'weight_decay': self.config['weight_decay'],
            })
            logger.info(f"Backbone parameters: {sum(p.numel() for p in backbone_params):,}")
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': self.config['lr'],
                'name': 'head',
                'weight_decay': self.config['weight_decay'],
            })
            logger.info(f"Head parameters: {sum(p.numel() for p in head_params):,}")
        
        # Create optimizer
        if self.config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        logger.info(f"Optimizer created: {self.config['optimizer']}")
        return optimizer
    
    def _log_to_wandb(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        #Log metrics to WandB with proper prefixing.
        if self.wandb_run is None or not distributed.is_main_process():
            return
        
        # Add prefix to all metrics
        wandb_dict = {f"{prefix}/{key}": value for key, value in metrics.items()}
        self.wandb_run.log(wandb_dict, step=step)
    
    def train(self):
        #Main training loop with iteration-based training.
        # Load checkpoint if resuming
        if self.config.get('resume', False):
            last_checkpoint_dir = find_latest_checkpoint(self.ckpt_dir)
            if last_checkpoint_dir:
                logger.info(f"Found checkpoint: {last_checkpoint_dir}")
                self.start_iter = load_checkpoint(
                    last_checkpoint_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    strict_loading=False,
                ) + 1
                logger.info(f"Resuming from iteration {self.start_iter}")
        
        # Setup metric logger
        metrics_file = self.output_dir / "training_metrics.json"
        metric_logger = MetricLogger(delimiter="  ", output_file=str(metrics_file))
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6g}"))
        metric_logger.add_meter("wd", SmoothedValue(window_size=1, fmt="{value:.6g}"))
        
        # Disable automatic garbage collection
        gc.disable()
        gc.collect()
        
        logger.info(f"Starting training from iteration {self.start_iter}")
        
        self.model.train()
        iteration = self.start_iter
        
        # Training loop
        for data in metric_logger.log_every(
            self.train_loader,
            print_freq=10,
            header="Training",
            n_iterations=self.max_iter,
            start_iteration=self.start_iter,
        ):
            if iteration >= self.max_iter:
                break
            
            # Garbage collection every 150 iterations
            if (iteration + 1) % 150 == 0:
                logger.info("Garbage collection")
                gc.collect()
            
            # Get learning rates and weight decay from schedules
            lr = self.lr_schedule[iteration]
            wd = self.wd_schedule[iteration]
            backbone_lr = self.backbone_lr_schedule[iteration]
            
            # Apply schedules to optimizer
            apply_optim_scheduler(self.optimizer, lr, wd, backbone_lr)
            
            # Move data to device
            images = data['image'].to(self.device, non_blocking=True)
            masks = data['mask'].to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            
            # Compute loss
            losses = self.criterion(outputs, masks)
            total_loss = losses['total_loss']
            
            # Check for NaN loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"NaN or Inf loss detected at iteration {iteration}, skipping")
                continue
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            grad_norm = None
            if self.config.get('grad_clip', 0) > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config['grad_clip']
                )
                if iteration % 10 == 0:
                    metric_logger.update(grad_norm=grad_norm.item())
            
            self.optimizer.step()
            
            # Log metrics to MetricLogger
            metric_logger.update(lr=lr)
            metric_logger.update(wd=wd)
            metric_logger.update(backbone_lr=backbone_lr)
            metric_logger.update(total_loss=total_loss.item())
            
            # Log individual loss components
            for key, value in losses.items():
                if key != 'total_loss':
                    metric_logger.update(**{key: value.item()})
            
            # Log to WandB
            if iteration % self.wandb_log_freq == 0:
                wandb_metrics = {
                    'lr': lr,
                    'wd': wd,
                    'backbone_lr': backbone_lr,
                    'total_loss': total_loss.item(),
                }
                
                # Add all individual loss components
                for key, value in losses.items():
                    if key != 'total_loss':
                        wandb_metrics[key] = value.item()
                
                # Add gradient norm if available
                if grad_norm is not None:
                    wandb_metrics['grad_norm'] = grad_norm.item()
                
                # Add epoch information
                wandb_metrics['epoch'] = iteration / self.iter_per_epoch
                
                self._log_to_wandb(wandb_metrics, step=iteration, prefix="train")
            
            # Validation
            if self.eval_period > 0 and (iteration + 1) % self.eval_period == 0:
                val_metrics = self.validate(iteration)
                self.model.train()
                
                # Log validation metrics to WandB
                self._log_to_wandb(val_metrics, step=iteration, prefix="val")
                
                # Update best validation loss
                val_loss = val_metrics['total_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                    
                    # Save best checkpoint
                    save_checkpoint(
                        self.ckpt_dir / "best",
                        iteration=iteration,
                        model=self.model,
                        optimizer=self.optimizer,
                        overwrite=True,
                    )
                    
                    # Log best metric to WandB
                    if self.wandb_run is not None and distributed.is_main_process():
                        self.wandb_run.log({"val/best_loss": self.best_val_loss}, step=iteration)
            
            # Checkpointing
            if (iteration + 1) % self.checkpoint_period == 0:
                torch.cuda.synchronize()
                save_checkpoint(
                    self.ckpt_dir / str(iteration),
                    iteration=iteration,
                    model=self.model,
                    optimizer=self.optimizer,
                    overwrite=True,
                )
                # Keep only the last N checkpoints
                keep_last_n_checkpoints(self.ckpt_dir, self.max_checkpoints_to_keep)
                logger.info(f"Checkpoint saved at iteration {iteration}")
            
            iteration += 1
        
        # Final validation
        logger.info("Running final validation...")
        final_val_metrics = self.validate(iteration - 1)
        
        # Log final validation to WandB
        self._log_to_wandb(final_val_metrics, step=iteration - 1, prefix="val_final")
        
        # Save final checkpoint
        save_checkpoint(
            self.ckpt_dir / "final",
            iteration=iteration - 1,
            model=self.model,
            optimizer=self.optimizer,
            overwrite=True,
        )
        
        # Synchronize metrics across processes
        metric_logger.synchronize_between_processes()
        
        # Log final summary to WandB
        if self.wandb_run is not None and distributed.is_main_process():
            summary_metrics = {
                'final/best_val_loss': self.best_val_loss,
                'final/final_val_loss': final_val_metrics['total_loss'],
                'final/total_iterations': iteration,
                'final/total_epochs': iteration / self.iter_per_epoch,
            }
            self.wandb_run.log(summary_metrics)
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Final validation loss: {final_val_metrics['total_loss']:.4f}")
        
        # Finish WandB run
        if self.wandb_run is not None and distributed.is_main_process():
            self.wandb_run.finish()
        
        # Re-enable garbage collection
        gc.enable()
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    @torch.no_grad()
    def validate(self, iteration: int) -> Dict[str, float]:
        #Validate the model and return all metrics.
        self.model.eval()
        
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Validation [Iter {iteration}]"
        
        for data in metric_logger.log_every(self.val_loader, 10, header):
            # Move data to device
            images = data['image'].to(self.device, non_blocking=True)
            masks = data['mask'].to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            losses = self.criterion(outputs, masks)
            
            # Update metrics
            metric_logger.update(total_loss=losses['total_loss'].item())
            for key, value in losses.items():
                if key != 'total_loss':
                    metric_logger.update(**{key: value.item()})
        
        # Synchronize metrics
        metric_logger.synchronize_between_processes()
        
        # Collect all validation metrics
        val_metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
        val_loss = val_metrics['total_loss']
        logger.info(f"Validation [Iter {iteration}]: Loss = {val_loss:.4f}")
        
        # Log individual loss components
        for key, value in val_metrics.items():
            if key != 'total_loss':
                logger.info(f"  {key}: {value:.4f}")
        
        return val_metrics


def main():
    # Setup logging
    setup_logging(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Train DINOv3 + Mask2Former Segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint_path', type=str, help='Path to pretrained DINOv3 checkpoint (overrides config)')
    parser.add_argument('--dataset_path', type=str, help='Path to ImageNet-S dataset (overrides config)')
    parser.add_argument('--output_dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, help='WandB run name')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with command line arguments
    if args.checkpoint_path:
        config['checkpoint_path'] = args.checkpoint_path
    if args.dataset_path:
        config['dataset_path'] = args.dataset_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.resume:
        config['resume'] = True
    if args.use_wandb:
        config['use_wandb'] = True
    if args.wandb_project:
        config['wandb_project'] = args.wandb_project
    if args.wandb_run_name:
        config['wandb_run_name'] = args.wandb_run_name
    
    config['seed'] = args.seed
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Mask2FormerSegmentationTrainer(config)
    
    # Start training
    logger.info("Starting training...")
    results = trainer.train()
    
    # Log final results
    logger.info("Training results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
import argparse
import copy
import gc
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed
from torch.distributed._tensor import DTensor

import dinov3.distributed as distributed
from dinov3.checkpointer import (
    find_latest_checkpoint,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    load_checkpoint,
    register_dont_save_hooks,
    save_checkpoint,
)
from dinov3.configs import setup_config, setup_job, setup_multidistillation
from dinov3.data import (
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
    CombinedDataLoader,
)
from dinov3.logging import MetricLogger, setup_logging
from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay
from dinov3.train.multidist_meta_arch import MultiDistillationMetaArch
from dinov3.train.ssl_meta_arch import SSLMetaArch

# Import segmentation components
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead

# Import segmentation loss and dataset
from segmentation_loss import create_loss_function  # From provided loss functions
from imagenet_s_dataset import create_imagenet_s_dataloaders  # From provided dataset

assert torch.__version__ >= (2, 1)
torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
torch.backends.cudnn.benchmark = False  # True

logger = logging.getLogger("dinov3")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv3 training", add_help=add_help)
    parser.add_argument("--config-file", default="dinov3/configs/train/vitl_segmentation.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "--eval_pretrained_weights",
        type=str,
        default="",
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        default="./local_dino",
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument("--seed", default=0, type=int, help="RNG seed")
    parser.add_argument(
        "--benchmark-codebase",
        action="store_true",
        help="test the codebase for a few iters",
    )
    parser.add_argument("--test-ibot", action="store_true", help="test ibot")
    parser.add_argument("--profiling", action="store_true", help="do profiling")
    parser.add_argument("--dump-fsdp-weights", action="store_true", help="dump fsdp weights")
    parser.add_argument("--record_ref_losses", action="store_true", help="record reference losses")
    parser.add_argument("--ref_losses_path", default="", type=str)
    parser.add_argument("--multi-distillation", action="store_true", help="run multi-distillation")

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    if "schedules" in cfg:
        logger.info("Using schedules v2")
        return build_schedulers_v2(cfg)

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        trunc_extra=cfg.optim["schedule_trunc_extra"],
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH] = (
        0  # mimicking the original schedules
    )
    logger.info("Schedulers ready.")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def build_schedulers_v2(cfg):
    iter_per_epoch = cfg.train.OFFICIAL_EPOCH_LENGTH
    total_iterations = cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs
    logger.info(f"Total training iterations {total_iterations}")

    # LR scaling rules
    lr_peak = cfg.schedules.lr.peak
    lr_end = cfg.schedules.lr.end
    if cfg.optim.scaling_rule == "linear_wrt_256":
        lr_peak *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        lr_end *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        lr_peak *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        lr_end *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        logger.info(
            f"Scaling rule {cfg.optim.scaling_rule}, LR peak {cfg.schedules.lr.peak} -> {lr_peak}, LR end {cfg.schedules.lr.end} -> {lr_end}"
        )
    else:
        logger.info(f"No scaling rule for {cfg.optim.scaling_rule=}")

    lr = linear_warmup_cosine_decay(
        start=cfg.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=iter_per_epoch * cfg.schedules.lr.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.lr.cosine_epochs if "cosine_epochs" in cfg.schedules.lr else None
        ),
    )
    last_layer_lr = lr.copy()
    last_layer_lr[: iter_per_epoch * cfg.schedules.lr.freeze_last_layer_epochs] = 0
    weight_decay = linear_warmup_cosine_decay(
        start=cfg.schedules.weight_decay.start,
        peak=cfg.schedules.weight_decay.peak,
        end=cfg.schedules.weight_decay.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.weight_decay.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.weight_decay.cosine_epochs
            if "cosine_epochs" in cfg.schedules.weight_decay
            else None
        ),
    )
    momentum = linear_warmup_cosine_decay(
        start=cfg.schedules.momentum.start,
        peak=cfg.schedules.momentum.peak,
        end=cfg.schedules.momentum.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.momentum.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.momentum.cosine_epochs if "cosine_epochs" in cfg.schedules.momentum else None
        ),
    )
    teacher_temp = linear_warmup_cosine_decay(
        start=cfg.schedules.teacher_temp.start,
        peak=cfg.schedules.teacher_temp.peak,
        end=cfg.schedules.teacher_temp.end,
        warmup_iterations=iter_per_epoch * cfg.schedules.teacher_temp.warmup_epochs,
        total_iterations=total_iterations,
        cosine_iterations=(
            iter_per_epoch * cfg.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in cfg.schedules.teacher_temp
            else None
        ),
    )
    return lr, weight_decay, momentum, teacher_temp, last_layer_lr


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        if is_last_layer:
            param_group["lr"] = last_layer_lr * lr_multiplier
        else:
            param_group["lr"] = lr * lr_multiplier


def do_test(cfg, model, iteration, process_group, do_low_freq=False):
    # dump a sharded checkpoint
    eval_dir = Path(cfg.train.output_dir) / "eval" / str(iteration)
    if distributed.is_subgroup_main_process():
        eval_dir.mkdir(parents=True, exist_ok=True)
    if cfg.train.sharded_eval_checkpoint:
        ckpt_path = eval_dir / "sharded_teacher_checkpoint"
        if distributed.is_subgroup_main_process():
            ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        teacher_backbone = model.model_ema
        save_checkpoint(
            ckpt_dir=ckpt_path, iteration=iteration, model=teacher_backbone, overwrite=True, process_group=process_group
        )
        if not distributed.is_subgroup_main_process():
            return
    else:
        new_state_dict = model.model_ema.state_dict()
        for k, tensor in list(new_state_dict.items()):
            if isinstance(tensor, DTensor):
                new_state_dict[k] = tensor.full_tensor()
        if not distributed.is_subgroup_main_process():
            return
        # save teacher checkpoint
        ckpt_path = eval_dir / "teacher_checkpoint.pth"
        torch.save({"teacher": new_state_dict}, ckpt_path)
        logger.info("Saved eval checkpoint: %s", ckpt_path)


def build_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
):
    # Collate function
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    if cfg.multidistillation.enabled:
        assert cfg.multidistillation.global_batch_size % distributed.get_subgroup_size() == 0
        local_batch_size = cfg.multidistillation.global_batch_size // distributed.get_subgroup_size()
        dataloader_batch_size_per_gpu = (
            cfg.multidistillation.global_batch_size + (distributed.get_world_size() - 1)
        ) // distributed.get_world_size()
    else:
        local_batch_size = None  # will default to the standard local batch size matching the data batch size
        dataloader_batch_size_per_gpu = cfg.train.batch_size_per_gpu

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[cfg.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=local_batch_size,
    )
    batch_size = dataloader_batch_size_per_gpu
    num_workers = cfg.train.num_workers
    dataset_path = cfg.train.dataset_path
    dataset = make_dataset(
        dataset_str=dataset_path,
        transform=model.build_data_augmentation_dino(cfg),
        target_transform=lambda _: (),
    )

    if isinstance(dataset, torch.utils.data.IterableDataset):
        sampler_type = SamplerType.INFINITE
    else:
        sampler_type = SamplerType.SHARDED_INFINITE if cfg.train.cache_dataset else SamplerType.INFINITE

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=cfg.train.seed + start_iter + 1,
        sampler_type=sampler_type,
        sampler_advance=start_iter * dataloader_batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return data_loader


def build_segmentation_data_loader(cfg, start_iter):
    """Build segmentation data loader for ImageNet-S dataset."""
    train_loader, val_loader = create_imagenet_s_dataloaders(
        root_dir=cfg.segmentation.dataset_path,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        resolution=cfg.crops.global_crops_size,
    )
    return train_loader


def build_multi_resolution_data_loader_from_cfg(
    cfg,
    model,
    start_iter,
    seed=65537,
):
    global_crops_sizes = (
        [cfg.crops.global_crops_size] if isinstance(cfg.crops.global_crops_size, int) else cfg.crops.global_crops_size
    )
    local_crops_sizes = (
        [cfg.crops.local_crops_size] if isinstance(cfg.crops.local_crops_size, int) else cfg.crops.local_crops_size
    )
    loader_ratios = (
        [cfg.crops.global_local_crop_pairs_ratios]
        if type(cfg.crops.global_local_crop_pairs_ratios) in [int, float]
        else cfg.crops.global_local_crop_pairs_ratios
    )
    assert len(global_crops_sizes) == len(local_crops_sizes) == len(loader_ratios)

    loaders = []
    for increment, (global_crops_size_i, local_crops_size_i) in enumerate(
        zip(global_crops_sizes, local_crops_sizes)
    ):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.crops.global_crops_size = global_crops_size_i
        cfg_i.crops.local_crops_size = local_crops_size_i
        cfg_i.train.seed = cfg.train.seed + increment + 1
        loaders.append(build_data_loader_from_cfg(cfg=cfg_i, model=model, start_iter=start_iter))

    if len(loaders) == 1:
        data_loader = loaders[0]
    else:
        data_loader = CombinedDataLoader(
            loaders_with_ratios=zip(loaders, loader_ratios),
            batch_size=cfg.train.batch_size_per_gpu,
            combining_mode=0,
            seed=seed,
            name="MultiResDL",
        )
    return data_loader


def do_train(cfg, model, resume=False):
    process_subgroup = distributed.get_process_subgroup()
    ckpt_dir = Path(cfg.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    # Optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    if cfg.multidistillation.enabled:
        register_dont_save_hooks(
            model,
            dont_save=[k for k, _ in model.state_dict().items() if k.startswith("teacher")],
        )
    model.init_weights()
    
    # Initialize segmentation head
    if cfg.segmentation.enabled:
        logger.info("Initializing segmentation head")
        # Get backbone name from student architecture
        if hasattr(cfg.student, 'arch'):
            backbone_name = f"dinov3_{cfg.student.arch}{cfg.student.patch_size}"
        else:
            backbone_name = "dinov3_vitl16"  # default
            
        model.segmentation_model = build_segmentation_decoder(
            backbone_model=model.student['backbone'],
            backbone_name=backbone_name,
            decoder_type="m2f",
            hidden_dim=cfg.segmentation.hidden_dim,
            num_classes=cfg.segmentation.num_classes,
        )
        
        # Initialize segmentation loss
        model.segmentation_loss_fn = create_loss_function(
            num_classes=cfg.segmentation.num_classes,
            weight_class=cfg.segmentation.loss_weight_class,
            weight_mask=cfg.segmentation.loss_weight_mask,
            weight_dice=cfg.segmentation.loss_weight_dice,
        )
    
    start_iter = 0
    if resume and (last_checkpoint_dir := find_latest_checkpoint(ckpt_dir)):
        logger.info(f"Checkpoint found {last_checkpoint_dir}")
        loaded_iter = (
            load_checkpoint(
                last_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                strict_loading=False,
                process_group=process_subgroup,
            )
            
        )
        if(isinstance(loaded_iter, str)):
                import re
                match = re.search(r'\d+', loaded_iter)
                loaded_iter = int(match.group()) if match else 0
        start_iter = loaded_iter + 1
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    if cfg.multidistillation.enabled:
        global_batch_size = cfg.multidistillation.global_batch_size
    else:
        global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_world_size()

    # Build data loader
    data_loader = build_multi_resolution_data_loader_from_cfg(
        cfg=cfg,
        model=model,
        start_iter=start_iter,
    )
    
    # Build segmentation data loader if enabled
    if cfg.segmentation.enabled:
        seg_data_loader = build_segmentation_data_loader(cfg, start_iter)
        seg_data_iter = iter(seg_data_loader)

    # Metric logging
    logger.info("Starting training from iteration %d", start_iter)
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    # Manual garbage collection
    gc.disable()
    gc.collect()
    if distributed.is_main_process():
        import wandb
        wandb_run = wandb.init(project="dinov3", name="Training Run with Segmentation Loss")
        wandb_run.watch(model, log="all", log_freq=1000)
    # Training loop
    student = model.student
    iteration = start_iter
    consecutive_nan_count = 0
    for data in metric_logger.log_every(
        data_loader,
        print_freq=10,
        header="Training",
        n_iterations=max_iter,
        start_iteration=start_iter,
    ):
        it = iteration
        data["global_batch_size"] = global_batch_size
        if iteration > max_iter:
            return

        # Garbage collection (trigger manually so it happens on all ranks at the same time)
        if (iteration + 1) % 150 == 0:
            logger.info("Garbage collection")
            gc.collect()

        # Learning rates and other schedules
        lr = lr_schedule[it]
        wd = wd_schedule[it]
        mom = momentum_schedule[it]
        teacher_temp = teacher_temp_schedule[it]
        last_layer_lr = last_layer_lr_schedule[it]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # Forward backward
        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)
        
        # Add segmentation loss if enabled
        if cfg.segmentation.enabled and iteration >= cfg.segmentation.start_iteration:
            try:
                seg_data = next(seg_data_iter)
            except StopIteration:
                seg_data_iter = iter(seg_data_loader)
                seg_data = next(seg_data_iter)
            
            # Move segmentation data to GPU
            seg_images = seg_data['image'].cuda()
            seg_masks = seg_data['mask'].cuda()
            
            # Get features from student backbone for segmentation
            with torch.no_grad():
                # Get multi-scale features from student backbone
                student_features = model.student['backbone'](seg_images)
                
            # Forward through segmentation head
            seg_outputs = model.segmentation_model(student_features)
            
            # Compute segmentation loss
            seg_loss_dict = model.segmentation_loss_fn(seg_outputs, seg_masks)
            seg_total_loss = seg_loss_dict['total_loss'] * cfg.segmentation.loss_weight
            
            # Add to total loss
            total_loss = total_loss + seg_total_loss
            
            # Add segmentation metrics
            metrics_dict.update({
                f"seg_{k}": v for k, v in seg_loss_dict.items()
            })
            metrics_dict["seg_total_weighted_loss"] = seg_total_loss

        loss_dict = metrics_dict

        # Organize losses into separate graphs using different prefixes
        wandb_log_dict = {}

        # DINO Loss Graph
        if "dino_local_crops_loss" in loss_dict:
            dino_loss_val = loss_dict["dino_local_crops_loss"].item() if hasattr(loss_dict["dino_local_crops_loss"], 'item') else loss_dict["dino_loss"]
            wandb_log_dict["dino/loss"] = dino_loss_val

        # KoLeo Loss Graph  
        if "koleo_loss" in loss_dict:
            koleo_loss_val = loss_dict["koleo_loss"].item() if hasattr(loss_dict["koleo_loss"], 'item') else loss_dict["koleo_loss"]
            wandb_log_dict["koleo/loss"] = koleo_loss_val

        # iBOT Loss Graph
        if "ibot_loss" in loss_dict:
            ibot_loss_val = loss_dict["ibot_loss"].item() if hasattr(loss_dict["ibot_loss"], 'item') else loss_dict["ibot_loss"]
            wandb_log_dict["ibot/loss"] = ibot_loss_val

        # Segmentation Loss Graphs
        if "seg_total_loss" in loss_dict:
            seg_total_loss_val = loss_dict["seg_total_loss"].item() if hasattr(loss_dict["seg_total_loss"], 'item') else loss_dict["seg_total_loss"]
            wandb_log_dict["segmentation/total_loss"] = seg_total_loss_val
            
        if "seg_class_loss" in loss_dict:
            seg_class_loss_val = loss_dict["seg_class_loss"].item() if hasattr(loss_dict["seg_class_loss"], 'item') else loss_dict["seg_class_loss"]
            wandb_log_dict["segmentation/class_loss"] = seg_class_loss_val
            
        if "seg_mask_loss" in loss_dict:
            seg_mask_loss_val = loss_dict["seg_mask_loss"].item() if hasattr(loss_dict["seg_mask_loss"], 'item') else loss_dict["seg_mask_loss"]
            wandb_log_dict["segmentation/mask_loss"] = seg_mask_loss_val
            
        if "seg_dice_loss" in loss_dict:
            seg_dice_loss_val = loss_dict["seg_dice_loss"].item() if hasattr(loss_dict["seg_dice_loss"], 'item') else loss_dict["seg_dice_loss"]
            wandb_log_dict["segmentation/dice_loss"] = seg_dice_loss_val

        # Log to wandb
        if distributed.is_main_process():
            wandb_run.log(wandb_log_dict, step=iteration)
            
        # Gradient clipping
        if cfg.optim.clip_grad:
            for k, v in student.items():
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    v.parameters(),
                    max_norm=cfg.optim.clip_grad,
                )
                metrics_dict[f"{k}_grad_norm"] = (
                    grad_norm.full_tensor().item()
                    if isinstance(grad_norm, torch.distributed.tensor.DTensor)
                    else grad_norm.item()
                )

        # Reduce total_loss to check for NaNs, reduce metrics for logging
        total_loss_all_ranks = total_loss.new_empty(distributed.get_subgroup_size())
        torch.distributed.all_gather_into_tensor(
            total_loss_all_ranks,
            total_loss.detach(),
            group=distributed.get_process_subgroup(),
        )
        total_loss = total_loss_all_ranks.mean()
        metrics_values = torch.stack(
            [torch.as_tensor(v, dtype=torch.float32, device=total_loss.device).detach() for v in metrics_dict.values()]
        )
        torch.distributed.all_reduce(
            metrics_values,
            op=torch.distributed.ReduceOp.AVG,
            group=distributed.get_process_subgroup(),
        )
        metrics_dict = dict(zip(metrics_dict.keys(), metrics_values))
        if total_loss_all_ranks.isnan().any():
            consecutive_nan_count += 1
            which_ranks = total_loss_all_ranks.isnan().nonzero().flatten().tolist()
            logger.warning("NaN loss detected on ranks: %s", which_ranks)
            logger.warning("Consecutive NaNs: %d", consecutive_nan_count)
            metrics_dict_str = "\n".join([f"{k}: {v}" for k, v in metrics_dict.items()])
            logger.warning("All-reduced metrics:\n%s", metrics_dict_str)
            if consecutive_nan_count > 2 and not cfg.multidistillation.enabled:
                msg = "Too many consecutive nans detected in loss, aborting..."
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            consecutive_nan_count = 0
        # Step optimizer
        optimizer.step()
        model.update_ema(mom)

        # Log metrics
        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(total_loss=total_loss, **metrics_dict)
        if distributed.is_main_process():
            wandb_run.log({"lr": lr, "wd": wd, "mom": mom, "last_layer_lr":last_layer_lr, "total_loss":total_loss})
        # Submit evaluation jobs
        if (
            cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
            # and iteration != max_iter - 1
        ):
            do_test(cfg, model, f"training_{iteration}", process_group=process_subgroup)
            torch.cuda.synchronize()

        # Checkpointing
        if (iteration + 1) % cfg.checkpointing.period == 0:
            torch.cuda.synchronize()
            save_checkpoint(
                ckpt_dir / str(iteration),
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                overwrite=True,
                process_group=process_subgroup,
            )
            if distributed.is_subgroup_main_process():
                keep_last_n_checkpoints(ckpt_dir, cfg.checkpointing.max_to_keep)
                if "keep_every" in cfg.checkpointing and (iteration + 1) % cfg.checkpointing.keep_every == 0:
                    keep_checkpoint_copy(ckpt_dir / str(iteration))

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(argv=None):
    if argv is None:
        args = get_args_parser().parse_args()
    else:
        args = get_args_parser().parse_args(argv[1:])
        args.output_dir = sys.argv[1]
    if args.multi_distillation:
        print("performing multidistillation run")
        cfg = setup_multidistillation(args)
        torch.distributed.barrier()
        logger.info("setup_multidistillation done")
        assert cfg.MODEL.META_ARCHITECTURE == "MultiDistillationMetaArch"
    else:
        print("train args")
        print(args)
        setup_job(output_dir=args.output_dir, seed=args.seed)
        cfg = setup_config(args, strict_cfg=False)
        logger.info(cfg)
        setup_logging(
            output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
            name="nan_logger",
        )
    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "MultiDistillationMetaArch": MultiDistillationMetaArch,
    }.get(cfg.MODEL.META_ARCHITECTURE, None)
    if meta_arch is None:
        raise ValueError(f"Unknown MODEL.META_ARCHITECTURE {cfg.MODEL.META_ARCHITECTURE}")
    logger.info(f"Making meta arch {meta_arch.__name__}")
    with torch.device("meta"):
        model = meta_arch(cfg)
    model.prepare_for_distributed_training()
    # Fill all values with `nans` so that we identify
    # non-initialized values
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device="cuda",
        ),
        recurse=True,
    )
    logger.info(f"Model after distributed:\n{model}")
    if args.eval_only:
        model.init_weights()
        iteration = (
            model.get_checkpointer_class()(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    main()