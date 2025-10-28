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

# Import custom modules
from simple_seg_head import DINOv3SegmentationModel
from imagenet_s_dataset import ImageNetSDataset
from seg_loss import create_loss_function

logger = logging.getLogger("dinov3")


def build_schedulers(cfg: Dict[str, Any], iter_per_epoch: int):
    """Build learning rate and weight decay schedules as numpy arrays."""
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
    backbone_lr_schedule = lr_schedule * 0.1
    
    logger.info("Schedulers ready.")
    return lr_schedule, wd_schedule, backbone_lr_schedule


def apply_optim_scheduler(optimizer, lr, wd, backbone_lr):
    """Apply learning rate and weight decay schedules to optimizer."""
    for param_group in optimizer.param_groups:
        is_backbone = param_group['name'] == 'backbone'
        param_group['weight_decay'] = wd
        if is_backbone:
            param_group['lr'] = backbone_lr
        else:
            param_group['lr'] = lr


class SegmentationTrainer:
    """Trainer class for DINOv3 segmentation on ImageNet-S."""
    
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
                project=config.get('wandb_project', 'dinov3-segmentation'),
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
        """Create and initialize the segmentation model."""
        model = DINOv3SegmentationModel(
            backbone_name=self.config['backbone_name'],
            num_classes=self.config['num_classes'],
            pretrained_weights=self.config.get('pretrained_weights', None),
        )
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _create_dataloaders(self) -> tuple:
        """Create train and validation dataloaders using DINOv3 patterns."""
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
        """Create the loss function."""
        return create_loss_function(
            loss_type=self.config['loss_type'],
            num_classes=self.config['num_classes'],
            weight_class=self.config.get('weight_class', 2.0),
            weight_mask=self.config.get('weight_mask', 5.0),
            weight_dice=self.config.get('weight_dice', 5.0),
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with separate parameter groups."""
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
                'lr': self.config['lr'] * 0.1,
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
        """Log metrics to WandB with proper prefixing."""
        if self.wandb_run is None or not distributed.is_main_process():
            return
        
        # Add prefix to all metrics
        wandb_dict = {f"{prefix}/{key}": value for key, value in metrics.items()}
        self.wandb_run.log(wandb_dict, step=step)
    
    def train(self):
        """Main training loop with iteration-based training."""
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
        """Validate the model and return all metrics."""
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
    
    parser = argparse.ArgumentParser(description='Train DINOv3 Segmentation Head')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
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
    trainer = SegmentationTrainer(config)
    
    # Start training
    logger.info("Starting training...")
    results = trainer.train()
    
    # Log final results
    logger.info("Training results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()