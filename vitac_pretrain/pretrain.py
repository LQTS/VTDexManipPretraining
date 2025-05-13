
import os
import sys

from einops import rearrange

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ["WANDB_API_KEY"] = 'eecc2d7f1d3232e500af4dded22bb3abdafdc11b'
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import hydra
import torch
import torch.distributed as dist
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from pathlib import Path

from vitac_core.util import LossLog, save_recons_tac, save_recons_imgs, visualize_recon_image
from vitac_core.conf import AcceleratorConfig, DatasetConfig, ModelConfig, TrackingConfig
from vitac_core.overwatch import OverwatchRich
from vitac_core.util import set_global_seed, do_resume, CheckpointSaver, Metrics
from vitac_core.util import ResumeableDistributedSampler
from vitac_core.models import get_model_optimizer
from vitac_core.datasets import get_datasets
# from pretrain_config import Pretrain_Config
from ConfigBank import Pretrain_Config

@dataclass
class PretrainConfig:
    # fmt: off
    defaults: List[Any] = field(default_factory=lambda: Pretrain_Config.model_dataset)  # DEFAULTS)
    hydra: Dict[str, Any] = field(default_factory=lambda: Pretrain_Config.hydra_)
    # recon pic save path
    recon_pic_save_path: str = Pretrain_Config.recon_pic_save_path

    # Command Line Arguments
    run_id: Optional[str] = None  # Run ID for Logging
    seed: int = 21  # Random Seed (for reproducibility)
    save_recons_freq: int = Pretrain_Config.save_recons_freq
    # Resume / Debug Behavior
    resume: bool = Pretrain_Config.resume  # Whether to resume an existing run...
    wandb_resume_id: Optional[str] = None  # W&B Run ID for `resume` behavior...

    # load model(checkpoint) to finetone
    if_finetone: bool = Pretrain_Config.if_finetone
    finetone_model_path: str = Pretrain_Config.finetone_model_path
    # Composable / Structured Arguments
    model: ModelConfig = MISSING  # Model architecture for pretraining
    dataset: DatasetConfig = MISSING  # List of datasets for pretraining
    accelerator: AcceleratorConfig = MISSING  # Accelerator (should always keep `torchrun`)
    tracking: TrackingConfig = MISSING  # Run/experiment tracking configuration
    # fmt: on


# Hydra Setup :: Retrieve ConfigStore (Singleton) & Register Components
cs = ConfigStore.instance()
cs.store(group="hydra/job_logging", name="overwatch_rich", node=OverwatchRich)
cs.store(name="config", node=PretrainConfig)

@hydra.main(config_path=None, config_name="config")
def pretrain(cfg: PretrainConfig) -> None:


    enable_distributed = cfg.accelerator.enable_distributed
    if enable_distributed:

        torch.multiprocessing.set_start_method("spawn", force=True)
        # Initialize Distributed Process Group --> assumes NCCL + Environment Variable Initialization (via `torchrun`)
        dist.init_process_group(backend="nccl", init_method="env://")
        device_id = dist.get_rank() % torch.cuda.device_count()
        is_rank_zero, rank, world_size = dist.get_rank() == 0, dist.get_rank(), dist.get_world_size()
    else:
        os.environ["LOCAL_RANK"] = '0'
        device_id = torch.device('cuda')
        is_rank_zero = True # 为主GPU
        rank = 0
        world_size = 1

    # Create Unique Run Name -- `resume = True` we assume the same "run_id"
    if cfg.run_id is None:
        cfg.run_id = run_dir = "{0}+{1}-ddp-x{2}".format(
      cfg.model.identifier +'-'+ cfg.model.remarks if cfg.model.remarks != "" else cfg.model.identifier,
            cfg.dataset.name +'-'+ cfg.dataset.remarks if cfg.dataset.remarks != "" else cfg.dataset.name,
            cfg.seed
        )
    else:
        run_dir = cfg.run_id

    # Setup Logging (Rank 0 Only!) and Directory Handling
    overwatch = logging.getLogger(__file__)
    overwatch.setLevel(logging.INFO if is_rank_zero else logging.ERROR)
    overwatch.info("VTDexManip Training :: Assembling the Legendary Defender...")
    if is_rank_zero:
        os.makedirs(run_dir, exist_ok=True)

    # Create Train Logger
    train_logger = LossLog()

    # Let's Get Started!
    overwatch.info(
        '\t=>> "Start to train."'
    )

    # Set Randomness & Get Dataloader `worker_init_fn` to ensure proper randomness in augmentations (if any)
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    # Initialize Model & Optimizer --> Wrap in DDP / Device Handling
    #   > Note :: For (Standard) DDP Training --> initializing Optimizer before DDP == initializing after!
    overwatch.info("Initializing Model, Optimizer, and Learning Rate Scheduler")
    model, optimizer, update_lr = get_model_optimizer(cfg.model, cfg.dataset, Path(run_dir) / "model_config.json")
    if enable_distributed:
        model = DDP(model.to(device_id), device_ids=[device_id], output_device=device_id)
    else:
        model = model.to(device_id)

    # Handle Resume / Checkpoint Loading
    resume_checkpoint, resume_epoch, resume_step = do_resume(cfg.resume, run_dir=run_dir)
    if cfg.if_finetone:
        # resume_checkpoint = cfg.finetone_model_path
        # resume_epoch = 0
        if enable_distributed:
            resume_state = torch.load(cfg.finetone_model_path, map_location=f"cuda:{device_id}")
        else:
            resume_state = torch.load(cfg.finetone_model_path)
        # load model params
        model.load_state_dict(resume_state[0], strict=False)

        if enable_distributed:
            dist.barrier()
    if resume_checkpoint is not None:
        # IMPORTANT --> Load weights by mapping specifically to `cuda:<device_id>`!
        if enable_distributed:
            resume_state = torch.load(resume_checkpoint, map_location=f"cuda:{device_id}")
        else:
            resume_state = torch.load(resume_checkpoint)

        state_dict = resume_state["model_state_dict"]
        if enable_distributed:
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            dist.barrier()
        else:
            if 'module' in list(state_dict.keys())[0]:
                # 去除 'module.' 前缀
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])



    # Create Checkpoint Saver and Save Initial Checkpoint
    saver = CheckpointSaver(cfg.tracking.checkpoint_strategy, run_dir, is_rank_zero=is_rank_zero)
    if (resume_checkpoint is None and resume_epoch == 0) or cfg.if_finetone:
        overwatch.info("  | Saving 0th Epoch Checkpoint (Model Initialization)")
        saver.save(
            epoch=resume_epoch, is_local_step=False, model=model, optimizer=optimizer, duration=0, train_loss=None, val_loss=None
        )
        if enable_distributed:
            dist.barrier()

    # Get Datasets --> Barrier after I/O Intensive Operation
    overwatch.info(f"Retrieving Dataset `{cfg.dataset.name}` prepared for Model `{cfg.model.arch}`")
    datasets_input_config = {
        "resolution": cfg.dataset.resolution,
        "normalization": cfg.dataset.normalization
    }
    if "tactile" in cfg.dataset.data_formats:
        datasets_input_config["tactile_type"] = cfg.model.tactile_type
    train_dataset, val_dataset = get_datasets(
        resume_epoch,  # 调试看是否为0
        cfg.dataset.name,
        cfg.model.arch,
        cfg.dataset.artifact_path,
        cfg.model.data_modality,
        cfg.dataset.data_formats,
        datasets_input_config,
    )

    if enable_distributed:
        dist.barrier()

    # Create Metrics =>> Handles on-the-fly computation, logging to JSONL and Weights & Biases
    metrics = Metrics(
        active_loggers=cfg.tracking.active_loggers,
        run_id=cfg.run_id,
        hparams=OmegaConf.to_container(cfg),
        model_arch=cfg.model.arch,
        is_rank_zero=is_rank_zero,
        tracking_cfg=cfg.tracking,
        tags=cfg.tracking.tags,
        resume=cfg.resume,
        resume_id=cfg.wandb_resume_id,
    )
    if enable_distributed:
        dist.barrier()

    # Configure Gradient Accumulation --> function of `effective_bsz`, `native_bsz`, and `WORLD_SIZE`
    assert cfg.model.effective_bsz % cfg.model.native_bsz == 0, "Device `native_bsz` must evenly divide `effective_bsz`"
    accumulate_grad_batches = cfg.model.effective_bsz // cfg.model.native_bsz // world_size
    overwatch.info(f"Running `{cfg.model.identifier}` Model Pretraining with Parameters =>")
    overwatch.info(f"  | Effective Batch Size = `{cfg.model.effective_bsz}`")
    overwatch.info(f"  | Per-Device Batch Size = `{cfg.model.native_bsz}`")
    overwatch.info(f"  | Distributed World Size = `{world_size}`")
    overwatch.info(f"  | Accumulation Steps = `{accumulate_grad_batches}`")

    # Start Train Loop --> Iterate through Epochs (Evaluation at end of Epoch)
    overwatch.info("Starting Training Loop")
    for epoch in range(resume_epoch, cfg.dataset.max_epochs):
        overwatch.info(f"  | [Epoch {epoch:03d}] Building Distributed Sampler & DataLoaders")
        train_dataset.set_epoch(epoch)
        if enable_distributed:
            dist.barrier()

        # [Custom] ResumeableDistributedSampler operates over *examples* --> start_step (full batches) * effective_bsz
        train_sampler = ResumeableDistributedSampler(
            seen_examples=resume_step * cfg.model.effective_bsz,
            resume_epoch=resume_epoch,
            dataset=train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.seed,
        )
        train_sampler.set_epoch(epoch)
        val_sampler = DistributedSampler(val_dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=False,
                                         drop_last=True)

        # Create Epoch DataLoaders
        train_dl = DataLoader(
            train_dataset,
            batch_size=cfg.model.native_bsz,
            sampler=train_sampler,
            shuffle=False,
            num_workers=cfg.accelerator.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=4,
            worker_init_fn=worker_init_fn,
        )
        val_dl = DataLoader(
            val_dataset, batch_size=cfg.model.native_bsz, sampler=val_sampler, shuffle=False, num_workers=cfg.accelerator.num_workers
        )

        # Book-Keeping =>> Set LR when `resume = True` (or starting from scratch)
        if (epoch == resume_epoch or epoch == 0) and cfg.if_finetone is not True:
            metrics.resume_time = (
                int(re.search("-t=(.+?).pt", str(resume_checkpoint)).group(1)) if resume_checkpoint is not None else 0
            )
            metrics.commit(
                global_step=resume_step + ((len(train_dataset) // cfg.model.effective_bsz) * resume_epoch),
                lr=update_lr(resume_epoch, resume_step / (len(train_dataset) // cfg.model.effective_bsz)),
                update_step_time=True,
            )

        # === Train Epoch ===
        model.train()
        status = metrics.get_status(epoch)
        save_or_not = False
        overwatch.info(f"  | [Epoch {epoch:03d}] Running Train Loop")
        with tqdm(
                total=len(train_dl) // accumulate_grad_batches, desc=status, leave=False, disable=not is_rank_zero
        ) as progress:

            for train_idx, batch in enumerate(train_dl):
                epoch_data = dict()
                # mask_patches = None
                # Model-Specific Handling
                loss, img, img_recons,  tac, tac_recons, mask_patches = forward_model(cfg, model, epoch_data, batch, device_id, metrics, is_train=True)

                save_or_not = recons_modalities(cfg, enable_distributed, epoch, img, img_recons, mask_patches, model, save_or_not, tac, tac_recons, "train")

                # Commit Loss (Prior to Normalization)
                metrics.commit(loss=loss)

                # Normalize Loss to account for Gradient Accumulation --> Backward!
                normalized_loss = loss / accumulate_grad_batches
                normalized_loss.backward()

                # Step =>> Check if done w/ Gradient Accumulation
                if (train_idx + 1) % accumulate_grad_batches == 0:
                    metrics.commit(update_step_time=True)

                    # Push Metrics every `log_frequency` steps...
                    if (metrics.global_step % cfg.tracking.log_frequency == 0) and cfg.if_finetone is not True:
                        status = metrics.push(epoch)

                    # Optimizer Step --> Increment Global Step, Learning Rate, and Checkpoint (if specified)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr = update_lr(
                        epoch,
                        (resume_step + ((train_idx + 1) // accumulate_grad_batches))
                        / (len(train_dataset) // cfg.model.effective_bsz),
                    )
                    metrics.commit(global_step=metrics.global_step + 1, lr=lr)
                    if (epoch + 1) % 10 == 0:
                        saver.save(
                            epoch,
                            is_local_step=True,
                            model=model,
                            optimizer=optimizer,
                            duration=int(time.time() - metrics.start_time) + metrics.resume_time,
                            local_step=resume_step + ((train_idx + 1) // accumulate_grad_batches),
                        )

                    # Update Progress Bar
                    progress.update()
                    progress.set_description(status)

        train_logger.log_epoch(epoch_data)
        # === After Train Epoch --> Clear Gradients and reset `resume_step` ===
        optimizer.zero_grad()
        resume_step = 0

        # === Validation ===
        overwatch.info(f"  | [Epoch {epoch:03d}] Running Validation Loop")
        model.eval()

        # Accumulate `validation_losses` in order to `all_reduce` later!
        val_losses = []
        epoch_data_val = dict()
        save_or_not = False
        with torch.no_grad():
            for batch in tqdm(val_dl, leave=False, disable=not is_rank_zero):
                # Model-Specific Handling
                # mask_patches = None
                val_loss, img, img_recons, tac, tac_recons, mask_patches = forward_model(cfg, model, epoch_data_val, batch, device_id, metrics, is_train=False)
                # save recon pics
                save_or_not = recons_modalities(cfg, enable_distributed, epoch, img, img_recons, mask_patches, model, save_or_not, tac, tac_recons, "val")


        # save val loss to local
        train_logger.log_epoch(epoch_data_val, is_val=True)
        # Add to Validation Losses
        val_losses.append(val_loss)
        # All Reduce --> Push Epoch Metrics --> Checkpoint!
        validation_loss = torch.stack(val_losses).mean()
        if enable_distributed:
            dist.all_reduce(validation_loss)
        avg_val_loss = validation_loss / world_size
        if is_rank_zero:
            epoch_status, train_loss, training_duration = metrics.push_epoch(epoch, avg_val_loss)
            if ((epoch + 1) % 10 == 0) or epoch == 4:
                saver.save(
                    epoch=epoch + 1,
                    is_local_step=False,
                    model=model,
                    optimizer=optimizer,
                    duration=training_duration,
                    train_loss=train_loss.item(),
                    val_loss=avg_val_loss.item(),
                )

        # === End of Epoch ===
        if enable_distributed:
            dist.barrier()

        # save logger
        train_logger.save_log(str(run_dir), extra_path='loss')

    # # save logger
    # train_logger.save_log(str(run_dir), extra_path='loss')

    # Finalize
    metrics.finalize()

    # And... we're done!
    overwatch.info("...and that's all, folks!")
    if enable_distributed:
        dist.barrier()

def forward_model(cfg, model, epoch_data, batch, device_id, metrics, is_train):
    mask_patches = None
    img_recons = None
    tac_recons = None
    img, tac = None, None
    if cfg.model.arch in {"vtt-cond", "vtt-condv2"}:
        img, tac = batch
        loss, img_recons, _ = model(
            img.to(device_id, non_blocking=True),
            tac.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )
        if is_train:
            metrics.commit(reconstruction_loss=loss)
        epoch_data["loss"] = loss
    elif cfg.model.arch in {"vtt-repic"}:
        img, tac = batch
        loss, img_recons, mask_patches = model(
            img.to(device_id, non_blocking=True),
            tac.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )
        epoch_data["loss"] = loss
        if is_train:
            metrics.commit(reconstruction_loss=loss)
    elif cfg.model.arch in {"v-repic"}:
        img, _ = batch
        loss, img_recons, mask_patches = model(
            img.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )
        epoch_data["loss"] = loss
        if is_train:
            metrics.commit(reconstruction_loss=loss)
    elif cfg.model.arch in {"v-repic-cl"}:
        img, _, _ = batch
        img = rearrange(img, "bsz n_frames C H W -> (bsz n_frames) C H W", )
        loss, img_loss, cl_loss, img_recons, mask_patches = model(
            img.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )

        epoch_data["loss"] = loss
        epoch_data["img_loss"] = img_loss
        # epoch_data["tac_loss"] = tac_loss
        epoch_data["cl_loss"] = cl_loss
        if is_train:
            metrics.commit(
                # loss=loss,
                img_reconstruction_loss=img_loss,
                # tac_reconstruction_loss=tac_loss,
                contrastive_learning_loss=cl_loss
            )
    elif cfg.model.arch in {"vtt-retac", "v-retac"}:
        img, tac = batch
        loss, tac_recons, _ = model(
            img.to(device_id, non_blocking=True),
            tac.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )
        epoch_data["loss"] = loss
        if is_train:
            metrics.commit(reconstruction_loss=loss)
    elif cfg.model.arch in {"t-retac"}:
        _, tac = batch
        loss, tac_recons, _ = model(
            tac.to(device_id, non_blocking=True)
        )
        epoch_data["loss"] = loss
        if is_train:
            metrics.commit(reconstruction_loss=loss)
    elif cfg.model.arch in {"vtt-reall", "v-reall", "vtt-reall-maskloss"}:
        img, tac = batch
        (
            loss, [img_recons, tac_recons],
            [img_loss, tac_loss], mask_patches
        ) = model(
            img.to(device_id, non_blocking=True),
            tac.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )
        epoch_data["loss"] = loss
        epoch_data["img_loss"] = img_loss
        epoch_data["tac_loss"] = tac_loss
        if is_train:
            metrics.commit(
                reconstruction_loss=loss,
                img_reconstruction_loss=img_loss,
                tac_reconstruction_loss=tac_loss
            )
    elif cfg.model.arch == "vt-reall-vtm":
        img, tac = batch
        (
            loss, [img_recons, tac_recons],
            [img_loss, tac_loss, matching_loss], mask_patches
        ) = model(
            img.to(device_id, non_blocking=True),
            tac.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )
        epoch_data["loss"] = loss
        epoch_data["img_loss"] = img_loss
        epoch_data["tac_loss"] = tac_loss
        epoch_data["matching_loss"] = matching_loss
        if is_train:
            metrics.commit(
                reconstruction_loss=loss,
                img_reconstruction_loss=img_loss,
                tac_reconstruction_loss=tac_loss,
                matching_loss=matching_loss
            )
    elif cfg.model.arch in {"vttext-reall"}:
        img, tac, lang, lang_mask = batch
        (
            loss, [img_recons, tac_recons],
            [img_loss, tac_loss], mask_patches
        ) = model(
            img.to(device_id, non_blocking=True),
            tac.to(device_id, non_blocking=True),
            lang.to(device_id, non_blocking=True),
            lang_mask.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )
        epoch_data["loss"] = loss
        epoch_data["img_loss"] = img_loss
        epoch_data["tac_loss"] = tac_loss
        if is_train:
            metrics.commit(
                reconstruction_loss=loss,
                img_reconstruction_loss=img_loss,
                tac_reconstruction_loss=tac_loss
            )

    elif cfg.model.arch in {"vt-reall-cl"}:
        img, tac, sim_label = batch
        tac = rearrange(tac, "bsz n_frames n_parts n_fingers -> (bsz n_frames) (n_parts n_fingers)", )
        img = rearrange(img, "bsz n_frames C H W -> (bsz n_frames) C H W", )
        (
            loss, [img_recons, tac_recons],
            [img_loss, tac_loss, cl_loss], mask_patches
        ) = model(
            img.to(device_id, non_blocking=True),
            tac.to(device_id, non_blocking=True),
            sim_label.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )
        epoch_data["loss"] = loss
        epoch_data["img_loss"] = img_loss
        epoch_data["tac_loss"] = tac_loss
        epoch_data["cl_loss"] = cl_loss
        if is_train:
            metrics.commit(
                reconstruction_loss=loss,
                img_reconstruction_loss=img_loss,
                tac_reconstruction_loss=tac_loss,
                contrastive_learning_loss=cl_loss
            )
    elif cfg.model.arch in {"vt-reall-tcl"}:
        img, tac, sim_label = batch
        tac = rearrange(tac, "bsz n_frames n_parts n_fingers -> (bsz n_frames) (n_parts n_fingers)", )
        img = rearrange(img, "bsz n_frames C H W -> (bsz n_frames) C H W", )
        (
            loss, [img_recons, tac_recons],
            [img_loss, tac_loss, tcn_loss, l1_loss, l2_loss, tcn_acc],
            mask_patches
        ) = model(
            img.to(device_id, non_blocking=True),
            tac.to(device_id, non_blocking=True),
            sim_label.to(device_id, non_blocking=True),
            mask_ratio=cfg.model.mask_ratio
        )
        epoch_data["loss"] = loss
        epoch_data["img_loss"] = img_loss
        epoch_data["tac_loss"] = tac_loss
        epoch_data["tcn_loss"] = tcn_loss
        epoch_data["l1_loss"] = l1_loss
        epoch_data["l2_loss"] = l2_loss
        epoch_data["tcn_acc"] = tcn_acc
        if is_train:
            metrics.commit(
                reconstruction_loss=loss,
                img_reconstruction_loss=img_loss,
                tac_reconstruction_loss=tac_loss,
                tcn_loss=tcn_loss,
                l1_loss=l1_loss,
                l2_loss=l2_loss,
                tcn_accuracy=tcn_acc
            )
    elif cfg.model.arch == "vr3m-nolang":
        img, _, _ = batch
        # tac = rearrange(tac,"bsz n_frames n_parts n_fingers -> (bsz n_frames) (n_parts n_fingers)", )
        # img = rearrange(img, "bsz n_frames C H W -> (bsz n_frames) C H W", )
        (
            loss, tcn_loss, l1_loss, l2_loss, tcn_acc
        ) = model(
            img.to(device_id, non_blocking=True)
        )
        epoch_data["loss"] = loss
        epoch_data["tcn_loss"] = tcn_loss
        epoch_data["l1_loss"] = l1_loss
        epoch_data["l2_loss"] = l2_loss
        epoch_data["tcn_acc"] = tcn_acc
        if is_train:
            metrics.commit(
                # loss=loss,
                tcn_loss=tcn_loss,
                l1_loss=l1_loss,
                l2_loss=l2_loss,
                tcn_accuracy=tcn_acc
            )
    else:
        raise ValueError(f"Forward() for Model `{cfg.model.arch}` is not implemented!")


    return loss, img, img_recons , tac, tac_recons, mask_patches

def recons_modalities(cfg, enable_distributed, epoch, img, img_recons, mask_patches, model, save_or_not, tac, tac_recons, save_dir):
    # save recon pictures
    if epoch % cfg.save_recons_freq == 0 and not save_or_not:

        if cfg.model.arch in {"vtt-reall","vt-reall-vtm", "vt-reall-cl", "vttext-reall", "v-reall", "vtt-retac", "v-retac", "t-retac",
                              "vtt-reall-maskloss", "vt-reall-tcl"}:
            save_or_not = True
            # save reconstruct tactile
            save_recons_tac(tac.cpu(), tac_recons.cpu(),
                            Path(cfg.recon_pic_save_path),
                            identify=f"{save_dir}_tac/{epoch}/", )
        if cfg.model.arch in {"v-repic","vt-reall-vtm", "v-repic-cl", "vttext-reall", "vt-reall-cl", "vtt-repic", "vtt-reall",
                              "v-reall", "vtt-reall-maskloss", "vt-reall-tcl"}:
            save_or_not = True
            # save reconstruct img
            # if not enable_distributed:
            #     generate_origin_img = model.generate_origin_img
            # else:
            #     generate_origin_img = model.module.generate_origin_img
            # [recon_imgs, mask_imgs] = generate_origin_img(img_recons.cpu(), img.cpu(),
            #                                                 mask_patches=mask_patches.cpu() if mask_patches is not None else None)
            #
            # save_recons_imgs(img.cpu(), recon_imgs.cpu(),
            #                  Path(cfg.recon_pic_save_path),
            #                  identify=f"train_pic/{epoch}/",
            #                  online_normalization=cfg.dataset.normalization,
            #                  mask_imgs=mask_imgs.cpu() if mask_imgs is not None else None),
            if not enable_distributed:
                model4vis = model
            else:
                model4vis = model.module
            visualize_recon_image(model=model4vis,
                                  save_dir=Path(cfg.recon_pic_save_path) / f'{save_dir}_pic' / str(epoch),
                                  img=img,
                                  img_recon=img_recons.cpu(),
                                  mask=mask_patches)
        if not save_or_not and cfg.model.arch != "vr3m-nolang":
            raise ValueError(f"Save Train Recons for Model `{cfg.model.arch}` is not implemented!")

    return save_or_not

if __name__ == "__main__":
    # General Defaults --> should use Tensor Cores (kinda) if you have them!
    torch.set_float32_matmul_precision("high")

    pretrain()