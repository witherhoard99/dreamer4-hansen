# train_tokenizer.py
import os
import time
import math
import random
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler

import wandb

from mp4_dataset import Mp4FrameDataset
from device_preprocessor import DevicePreprocessor
from task_set import TASK_SET
from sharded_frame_dataset import ShardedFrameDataset
from model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, temporal_unpatchify,
    recon_loss_from_mae, lpips_on_mae_recon,
)

try:
    import lpips
except ImportError:
    lpips = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def seed_everything(seed: int):
    rank = int(os.environ.get("RANK", "0"))
    s = (int(seed) + rank) % (2 ** 32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    seed_everything(info.seed)


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


class LossNormalizer(nn.Module):
    """
    Normalizes losses by their root-mean-square (RMS) value using an EMA.
    This helps balance the scale of MSE vs LPIPS automatically.
    """

    def __init__(self, num_losses=1, beta=0.99, eps=1e-8):
        super().__init__()
        self.register_buffer('exp_avg_sq', torch.ones(num_losses))
        self.beta = beta
        self.eps = eps

    def forward(self, losses: torch.Tensor, update_ema: bool = True):
        # losses should be a 1D tensor of shape (num_losses,)
        if update_ema:
            with torch.no_grad():
                self.exp_avg_sq.lerp_(losses.detach().square(), 1.0 - self.beta)

        rms = self.exp_avg_sq.sqrt().clamp(min=self.eps)
        return losses / rms


@torch.no_grad()
def log_tokenizer_viz_wandb(
    *,
    x_btchw: torch.Tensor,          # (B,T,C,H,W) float in [0,1]
    pred_btnd: torch.Tensor,        # (B,T,Np,Dp) float in [0,1]
    mae_mask_btNp1: torch.Tensor,   # (B,T,Np,1) bool True=masked
    patch: int,
    step: int,
    max_items: int = 8,
    max_T: int = 6,
    tag: str = "tokenizer/viz",
):
    B, T, C, H, W = x_btchw.shape
    Tv = min(T, max_T)
    Bv = min(B, max_items)

    # patchify target
    target_btnd = temporal_patchify(x_btchw[:, :Tv], patch)  # (B,Tv,Np,Dp)

    # panels (patch space)
    masked_input_btnd = torch.where(mae_mask_btNp1[:, :Tv], torch.zeros_like(target_btnd), target_btnd)
    recon_masked_btnd = torch.where(mae_mask_btNp1[:, :Tv], pred_btnd[:, :Tv], target_btnd)
    recon_full_btnd   = pred_btnd[:, :Tv]

    # to image space (B,T,C,H,W)
    target_img = temporal_unpatchify(target_btnd,       H, W, C, patch)
    masked_img = temporal_unpatchify(masked_input_btnd, H, W, C, patch)
    rmask_img  = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    rfull_img  = temporal_unpatchify(recon_full_btnd,   H, W, C, patch)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        # (B,T,C,H,W) -> (B,C,H,T*W)
        x = x[:, :Tv]
        return x.permute(0, 2, 3, 1, 4).contiguous().view(x.shape[0], C, H, Tv * W)

    tgt = tile_time(target_img[:Bv])
    msk = tile_time(masked_img[:Bv])
    rm  = tile_time(rmask_img[:Bv])
    rf  = tile_time(rfull_img[:Bv])

    panel = torch.cat([tgt, msk, rm, rf], dim=2)  # (Bv,C,4H,Tv*W)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)  # (C,Bv*4H,Tv*W)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {
            tag: wandb.Image(
                big_hwc,
                caption="rows=target/masked/recon_masked/recon_full",
            ),
            "tokenizer/masked_frac": float(mae_mask_btNp1[:, :Tv].float().mean().item()),
        },
        step=step,
    )


def save_ckpt(path: Path, *, step: int, epoch: int, model, opt, scheduler, scaler, normalizer, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "opt": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "normalizer": normalizer.state_dict() if normalizer is not None else None,
        "args": vars(args),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)

def load_ckpt(path: Path, *, model, opt, scheduler, scaler, normalizer) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"]
    (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=True)
    opt.load_state_dict(ckpt["opt"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if normalizer is not None and ckpt.get("normalizer") is not None:
        normalizer.load_state_dict(ckpt["normalizer"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def get_scheduler(optimizer, args):
    if args.lr_scheduler == "none":
        return None

    grad_accum = max(1, int(args.grad_accum))
    max_updates = args.max_steps // grad_accum
    warmup_updates = args.lr_warmup_steps // grad_accum

    def lr_lambda(step):
        # Warmup
        if step < warmup_updates:
            return float(step) / float(max(1, warmup_updates))

        # After warmup
        progress = float(step - warmup_updates) / float(max(1, max_updates - warmup_updates))
        progress = min(1.0, max(0.0, progress))

        if args.lr_scheduler == "linear":
            # Linear decay to lr_min / lr
            return max(args.lr_min / args.lr, 1.0 -  progress * (1.0 - args.lr_min / args.lr))

        elif args.lr_scheduler == "cosine":
            # Cosine decay to lr_min / lr
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.lr_min / args.lr + (1.0 - args.lr_min / args.lr) * cosine_decay

        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_parameter_groups(model, weight_decay: float):
    decay = []
    no_decay = []

    # Keywords matching the 2D latents and embeddings in model.py
    no_decay_keywords = [
        "latents",  # Encoder.latents
        "patch_queries",  # Decoder.patch_queries
        "register_tokens",  # Dynamics.register_tokens (if used later)
        "agent_base",  # TaskEmbedder.agent_base (if used later)
        "embed",  # Catches nn.Embedding weights
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1D tensors catch ALL biases, RMSNorm scales, and the 1D mask_token
        # The keyword check catches your explicit 2D structural tokens
        if param.ndim <= 1 or any(kw in name for kw in no_decay_keywords):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]

def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed + rank)

    dataset = Mp4FrameDataset(
        dir=args.data_dirs,
        seq_len=args.seq_len,
        iid_sampling=True,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    # ---- model ----
    assert args.H % args.patch == 0 and args.W % args.patch == 0
    n_patches = (args.H // args.patch) * (args.W // args.patch)
    d_patch = args.patch * args.patch * args.C

    assert args.d_model % args.n_heads == 0, "d_model must be divisible by n_heads"

    loss_normalizer = None
    if args.use_loss_norm:
        loss_normalizer = LossNormalizer(num_losses=2).to(device)

    enc = Encoder(
        patch_dim=d_patch,
        d_model=args.d_model,
        n_latents=args.n_latents,
        n_patches=n_patches,
        n_heads=args.n_heads,
        depth=args.depth,
        d_bottleneck=args.d_bottleneck,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        mae_p_min=args.mae_p_min,
        mae_p_max=args.mae_p_max,
        gradient_checkpointing=args.grad_checkpointing,
        qk_norm=args.qk_norm,
        soft_cap=args.soft_cap
    )
    dec = Decoder(
        d_bottleneck=args.d_bottleneck,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        n_latents=args.n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        gradient_checkpointing=args.grad_checkpointing,
        qk_norm=args.qk_norm,
        soft_cap=args.soft_cap
    )
    model = Tokenizer(enc, dec).to(device)

    def preprocessor_fn(batch: torch.Tensor):
        nonlocal args
        assert isinstance(batch, torch.Tensor)

        batch = batch.to(torch.float32).div_(255.0)
        patch = temporal_patchify(batch, args.patch)
        return batch, patch

    if is_rank0():
        print(model)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Learnable parameters: {param_count:,}")

    if args.compile:
        model = torch.compile(model)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    param_groups = get_parameter_groups(model, args.weight_decay)
    opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)

    scheduler = get_scheduler(opt, args)
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(device="cuda", enabled=False)

    preprocessor = DevicePreprocessor(loader=loader, device=device, preprocess_fn=preprocessor_fn)

    # ---- lpips ----
    if args.lpips_weight > 0.0:
        assert lpips is not None, "pip install lpips"
        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_fn.eval()
        lpips_fn.requires_grad_(False)

        lpips_fn = torch.compile(lpips_fn)
    else:
        lpips_fn = None

    # ---- wandb ----
    if is_rank0():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode="online",
            config=vars(args),
        )

    # ---- resume ----
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch = load_ckpt(
            Path(args.resume), model=model,
            opt=opt, scheduler=scheduler,
            scaler=scaler, normalizer=loss_normalizer
        )

        if is_rank0():
            print(f"[rank0] Resumed from {args.resume} (samples_seen={step * args.batch_size * world_size}, epoch={start_epoch})")

    # ---- train ----
    model.train()
    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))

    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_vram_bytes / (1024 ** 3)

    if (total_vram_gb > 16):
        raise RuntimeError("Update the logging code so that step is a normal step (just step++). "
                           "Then, use samples_seen in the wandb viewer. "
                           "This will break backwards compat viewing with code on 5070ti but oh well"
                           "Also changing batch_size cooks comparability anyway because it changes the lr schedule"
                           "this is because the script takes in max_steps, not max_samples"
                           "ALSO EXPERIMENT WITH INCREASED LR. It should be fine with things like QK Norm")

    while step < args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for x, patches in preprocessor:
                if step >= args.max_steps:
                    break

                do_step = ((step + 1) % grad_accum == 0)

                if ddp and not do_step:
                    sync_context = model.no_sync()
                else:
                    sync_context = nullcontext()

                with sync_context:
                    with autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
                        with torch.no_grad():
                            if is_rank0() and step % args.log_every == 0:
                                z, _ = (model.module.encoder if hasattr(model, "module") else model.encoder)(patches)
                                wandb.log({"debug/z_std": float(z.float().std().item())}, step=(step * args.batch_size * world_size))

                        pred, mae_mask, keep_prob = model(patches)

                    # losses in fp32 (outside autocast)
                    mse = recon_loss_from_mae(pred, patches, mae_mask)
                    if lpips_fn is not None and args.lpips_weight > 0.0:
                        lp = lpips_on_mae_recon(
                            lpips_fn, pred, patches, mae_mask,
                            H = args.H, W = args.W, C = args.C, patch = args.patch,
                            subsample_frac = args.lpips_frac
                        )
                    else:
                        lp = torch.zeros((), device=device)

                    # --- Normalization Logic ---
                    mse_unnorm, lp_unnorm = mse.detach().clone(), lp.detach().clone()
                    mse_final, lp_final = mse, lp

                    if loss_normalizer is not None:
                        # Stack losses to process through normalizer
                        combined = torch.stack([mse, lp])

                        if ddp:
                            combined_synced = combined.detach().clone()
                            dist.all_reduce(combined_synced)
                            combined_synced /= world_size
                        else:
                            combined_synced = combined.detach()

                        # 1. Update the EMA using the globally synced tensor
                        if model.training:
                            _ = loss_normalizer(combined_synced, update_ema=True)

                        # 2. Get loss using local tensor
                        normed = loss_normalizer(combined, update_ema=False)

                        mse_final, lp_final = normed[0], normed[1]

                    loss = mse_final + args.lpips_weight * lp_final

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at step {step}: loss={loss} mse={mse} lp={lp}")

                    loss_to_backprop = loss / grad_accum

                    scaler.scale(loss_to_backprop).backward()

                # capture grad norm if clipping is enabled
                grad_norm = None

                if do_step:
                    if args.grad_clip > 0:
                        # unscale before clipping when using AMP (no-op if scaler disabled)
                        if use_amp:
                            scaler.unscale_(opt)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            (model.module if hasattr(model, "module") else model).parameters(),
                            max_norm=args.grad_clip,
                        )

                    if use_amp:
                        scaler.step(opt)

                        if is_rank0() and step % args.log_every == 0:
                            wandb.log({"amp/scale": float(scaler.get_scale())}, step=(step * args.batch_size * world_size))

                        scaler.update()
                    else:
                        opt.step()

                    if scheduler is not None:
                        scheduler.step()

                    opt.zero_grad(set_to_none=True)

                # ---- logging ----
                if is_rank0() and (step % args.log_every == 0):
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    log_dict = {
                        "loss/total": float(loss.item()),
                        "loss/mse_prenorm": float(mse_unnorm.item()),
                        "loss/lpips_prenorm": float(lp_unnorm.item()),
                        "stats/psnr": float(psnr.item()),
                        "stats/keep_prob": float(keep_prob.mean().item()),
                        "stats/masked_frac": float(mae_mask.float().mean().item()),
                        "lr": float(opt.param_groups[0]["lr"]),
                        "time/hrs": (time.time() - t0) / 3600.0,
                        "samples_seen": step * args.batch_size * world_size,
                    }

                    if loss_normalizer is not None:
                        log_dict.update({
                            "loss/mse_normed": float(mse_final.item()),
                            "loss/lpips_normed": float(lp_final.item()),
                            "norm/mse_rms": float(loss_normalizer.exp_avg_sq[0].sqrt().item()),
                            "norm/lpips_rms": float(loss_normalizer.exp_avg_sq[1].sqrt().item()),
                        })

                    if grad_norm is not None:
                        log_dict["stats/grad_norm"] = float(grad_norm)

                    wandb.log(log_dict, step=step * args.batch_size * world_size)

                if is_rank0() and (step % args.print_every == 0):
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| mse={mse.item():.6f} | lpips={lp.item():.4f} "
                        f"| psnr={psnr.item():.2f} | keep={keep_prob.mean().item():.3f}"
                    )

                # ---- viz ----
                if is_rank0() and args.viz_every > 0 and (step % args.viz_every == 0):
                    log_tokenizer_viz_wandb(
                        x_btchw=x,
                        pred_btnd=pred,
                        mae_mask_btNp1=mae_mask,
                        patch=args.patch,
                        step=(step * args.batch_size * world_size),
                        max_items=args.viz_max_items,
                        max_T=args.viz_max_T,
                    )

                # ---- ckpt ----
                if is_rank0() and args.save_every > 0 and ((step + 1) % (args.save_every * grad_accum) == 0):
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(
                        ckpt_path,
                        step=step, epoch=epoch,
                        model=model, opt=opt,
                        scheduler=scheduler, scaler=scaler,
                        args=args, normalizer=loss_normalizer
                    )
                    # also update a "latest" pointer
                    latest = ckpt_dir / "latest.pt"
                    save_ckpt(
                        latest,
                        step=step, epoch=epoch,
                        model=model, opt=opt,
                        scheduler=scheduler, scaler=scaler,
                        args=args, normalizer=loss_normalizer
                    )

                step += 1

            start_epoch = epoch + 1

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dirs", type=str, nargs="+", default=[   # paths to preprocessed frames
        "/<path>/expert-shards",
        "/<path>/mixed-small-shards",
        "/<path>/mixed-large-shards",
    ])
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)

    # image / patching
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--C", type=int, default=3)
    p.add_argument("--patch", type=int, default=4)

    # model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--n_latents", type=int, default=16)
    p.add_argument("--d_bottleneck", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)
    p.add_argument("--mae_p_min", type=float, default=0.0)
    p.add_argument("--mae_p_max", type=float, default=0.9)

    p.add_argument("--use_loss_norm", action="store_true", help="Use RMS loss normalization")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--qk_norm", action="store_true", help="Use QK normalization")
    p.add_argument("--soft_cap", type=float, default=0.0, help="Attention logit soft capping value (0.0 means disabled)")

    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "linear", "cosine"])
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--lr_min", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10_000_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # lpips
    p.add_argument("--lpips_weight", type=float, default=0.2)
    p.add_argument("--lpips_frac", type=float, default=0.5)
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    # logging / viz
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--print_every", type=int, default=100)
    p.add_argument("--viz_every", type=int, default=500)
    p.add_argument("--viz_max_items", type=int, default=4)
    p.add_argument("--viz_max_T", type=int, default=8)

    # wandb
    p.add_argument("--wandb_project", type=str, default="dreamer4-tokenizer")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/tokenizer_ckpts")
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--resume", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

    train(p.parse_args())
