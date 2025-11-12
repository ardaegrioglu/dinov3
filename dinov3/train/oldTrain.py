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

assert torch.__version__ >= (2, 1)
torch.backends.cuda.matmul.allow_tf32 = True  # pytorch 1.12 sets this to false by default
torch.backends.cudnn.benchmark = False  # True

# Import segmentation components
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead

# Import segmentation loss and dataset
from dinov3.eval.segmentation.models.newSegHead.seg_loss import create_loss_function  # From provided loss functions
from dinov3.eval.segmentation.models.newSegHead.imagenet_s_dataset import create_imagenet_s_dataloaders  # From provided dataset
logger = logging.getLogger("dinov3")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv3 training", add_help=add_help)
    parser.add_argument("--config-file", default="dinov3/configs/train/vitl_gram_anchor.yaml", metavar="FILE", help="path to config file")
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
    gram_teacher_crops_sizes = (
        [cfg.crops.gram_teacher_crops_size]
        if cfg.crops.gram_teacher_crops_size is None or isinstance(cfg.crops.gram_teacher_crops_size, int)
        else cfg.crops.gram_teacher_crops_size
    )
    loader_ratios = (
        [cfg.crops.global_local_crop_pairs_ratios]
        if type(cfg.crops.global_local_crop_pairs_ratios) in [int, float]
        else cfg.crops.global_local_crop_pairs_ratios
    )
    assert len(global_crops_sizes) == len(local_crops_sizes) == len(gram_teacher_crops_sizes) == len(loader_ratios)

    loaders = []
    for increment, (global_crops_size_i, local_crops_size_i, gram_teacher_crops_size_i) in enumerate(
        zip(global_crops_sizes, local_crops_sizes, gram_teacher_crops_sizes)
    ):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.crops.global_crops_size = global_crops_size_i
        cfg_i.crops.local_crops_size = local_crops_size_i
        cfg_i.crops.gram_teacher_crops_size = gram_teacher_crops_size_i
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

def build_segmentation_data_loader(cfg, start_iter):
    """Build segmentation data loader for ImageNet-S dataset."""
    print(cfg.segmentation)
    train_loader, val_loader = create_imagenet_s_dataloaders(
        root_dir='/mmfs1/gscratch/krishna/ardae/dinov3/ImageNet-S/Dataset/ImageNetS919',
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        resolution=cfg.crops.global_crops_size,
    )
    return train_loader



def do_train(cfg, model, resume=True):
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
        # ===== STEP 3: NOW add segmentation (after checkpoint loading) =====
    if cfg.segmentation.enabled:
        logger.info("Adding segmentation head AFTER checkpoint loading")
        
        # Get backbone name
        if hasattr(cfg.student, 'arch'):
            backbone_name = f"dinov3_{cfg.student.arch}{cfg.student.patch_size}"
        else:
            backbone_name = "dinov3_vitl16"
        
        # Build segmentation decoder
        model.segmentation_model = build_segmentation_decoder(
            backbone_model=model.student['backbone'],
            backbone_name=backbone_name,
            decoder_type="m2f",
            hidden_dim=cfg.segmentation.hidden_dim,
            num_classes=cfg.segmentation.num_classes,
        ).cuda()
        
        # Initialize segmentation loss
        model.segmentation_loss_fn = create_loss_function(
            num_classes=cfg.segmentation.num_classes,
            weight_class=cfg.segmentation.loss_weight_class,
            weight_mask=cfg.segmentation.loss_weight_mask,
            weight_dice=cfg.segmentation.loss_weight_dice,
        )
        
        # ===== STEP 4: Rebuild optimizer with segmentation parameters =====
        param_groups = model.get_params_groups()
        
        # Add segmentation parameters
        seg_param_group = {
            'params': model.segmentation_model.parameters(),
            'lr': cfg.optim.lr,
            'weight_decay': cfg.optim.weight_decay,
            'is_last_layer': False,
            'lr_multiplier': 1.0,
            'wd_multiplier': 1.0,
        }
        param_groups.append(seg_param_group)
        
        # Rebuild optimizer
        optimizer = build_optimizer(cfg, param_groups)
        
        logger.info(f"Segmentation model created with {sum(p.numel() for p in model.segmentation_model.parameters())} parameters")
        logger.info("Rebuilt optimizer to include segmentation parameters")
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
        wandb_run = wandb.init(project="dinov3", name="Training Run continuation close to gram loss kicking in")
        wandb_run.watch(model, log="all", log_freq=1000)
    # Training loop
    student = model.student
    iteration = start_iter
    num_gram_updates = 0
    if (
        cfg.gram.use_loss
        and model.has_gram_teacher
        and cfg.gram.rep_update
        and start_iter > 0
        and start_iter >= cfg.gram.it_first_update
    ):
        # If `start_iter == it_first_update`, we have performed one gram teacher update after
        # iteration `start_iter - 1`, except if we are starting training from scratch and `start_iter == 0`.
        num_gram_updates = math.ceil((start_iter + 1 - cfg.gram.it_first_update) / cfg.gram.update_frequency)
        logger.info(f"Gram was updated {num_gram_updates} times before iteration {start_iter}")
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

        if cfg.gram.use_loss and model.gram_it_load_ema_teacher == it:
            logger.info(f"Loading EMA teacher into Gram teacher before iteration {it}")
            model.gram_load_ema_teacher()

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
        loss_dict = metrics_dict

                
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

            # Forward through segmentation model
            seg_outputs = model.segmentation_model(seg_images)

            # ===== FIX: UPSAMPLE PREDICTIONS TO MATCH GROUND TRUTH RESOLUTION =====
            # Get the ground truth spatial size
            target_size = seg_masks.shape[-2:]  # (H, W) from seg_masks
    
            # Upsample predicted masks to match ground truth size
            seg_outputs["pred_masks"] = torch.nn.functional.interpolate(
                seg_outputs["pred_masks"],
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
    
            # Also upsample auxiliary outputs if they exist
            if "aux_outputs" in seg_outputs:
                for aux_output in seg_outputs["aux_outputs"]:
                    aux_output["pred_masks"] = torch.nn.functional.interpolate(
                        aux_output["pred_masks"],
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )

            # Compute segmentation loss (now sizes match!)
            seg_loss_dict = model.segmentation_loss_fn(seg_outputs, seg_masks)
            seg_total_loss = seg_loss_dict['total_loss'] * cfg.segmentation.loss_weight

            # Add to total loss
            total_loss = total_loss + seg_total_loss

            # Add segmentation metrics
            metrics_dict.update({
                f"seg_{k}": v for k, v in seg_loss_dict.items()
            })
            metrics_dict["seg_total_weighted_loss"] = seg_total_loss



        # Organize losses into separate graphs using different prefixes
        wandb_log_dict = {}

        # DINO Loss Graph
        if "dino_local_crops_loss" in loss_dict:
            dino_loss_val = loss_dict["dino_local_crops_loss"].item() if hasattr(loss_dict["dino_local_crops_loss"], 'item') else loss_dict["dino_loss"]
            wandb_log_dict["dino/loss"] = dino_loss_val
            # Add weight and weighted loss if you have access to model weights
            try:
                wandb_log_dict["dino/dino_local_loss_weight"] = loss_dict["dino_local_loss_weight"]
                wandb_log_dict["dino/weighted_loss"] = loss_dict["dino_local_loss_weight"] * dino_loss_val
            except AttributeError:
                pass

        # KoLeo Loss Graph  
        if "koleo_loss" in loss_dict:
            koleo_loss_val = loss_dict["koleo_loss"].item() if hasattr(loss_dict["koleo_loss"], 'item') else loss_dict["koleo_loss"]
            wandb_log_dict["koleo/loss"] = koleo_loss_val
            try:
                wandb_log_dict["koleo/weight"] = model.koleo_loss_weight
                wandb_log_dict["koleo/weighted_loss"] = model.koleo_loss_weight * koleo_loss_val
            except AttributeError:
                pass

        # iBOT Loss Graph
        if "ibot_loss" in loss_dict:
            ibot_loss_val = loss_dict["ibot_loss"].item() if hasattr(loss_dict["ibot_loss"], 'item') else loss_dict["ibot_loss"]
            wandb_log_dict["ibot/loss"] = ibot_loss_val
            try:
                wandb_log_dict["ibot/weight"] = model.ibot_loss_weight
                wandb_log_dict["ibot/weighted_loss"] = model.ibot_loss_weight * ibot_loss_val
            except AttributeError:
                pass

        # Gram Loss Graph (if enabled)
        if "gram_loss" in loss_dict:
            gram_loss_val = loss_dict["gram_loss"].item() if hasattr(loss_dict["gram_loss"], 'item') else loss_dict["gram_loss"]
            wandb_log_dict["gram/loss"] = gram_loss_val
    
            if "gram_loss_weight" in loss_dict:
                gram_weight = loss_dict["gram_loss_weight"]
                wandb_log_dict["gram/weight"] = gram_weight
                wandb_log_dict["gram/weighted_loss"] = gram_loss_val * gram_weight

        # Gram Stats Graph (if enabled)
        if "stats_only/masked_gram_loss" in loss_dict:
            wandb_log_dict["gram_stats/masked_loss"] = loss_dict["stats_only/masked_gram_loss"].item() if hasattr(loss_dict["stats_only/masked_gram_loss"], 'item') else loss_dict["stats_only/masked_gram_loss"]

        if "stats_only/unmasked_gram_loss" in loss_dict:
            wandb_log_dict["gram_stats/unmasked_loss"] = loss_dict["stats_only/unmasked_gram_loss"].item() if hasattr(loss_dict["stats_only/unmasked_gram_loss"], 'item') else loss_dict["stats_only/unmasked_gram_loss"]
            

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

        # [GRAM] Update gram teacher when using gram teacher and frequent updates
        if (
            cfg.gram.use_loss
            and model.gram_rep_update
            and (it + 1) >= model.gram_it_first_update
            and (it + 1) % model.gram_update_frequency == 0
            and (cfg.gram.max_updates is None or num_gram_updates < cfg.gram.max_updates)
        ):
            logger.info(f"Updating Gram teacher from EMA teacher after iteration {it}")
            model.update_gram()
            num_gram_updates += 1

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
        #breakpoint()
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
