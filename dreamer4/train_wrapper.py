import os, sys; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import torch
import task_set
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR))

TASK_NAME = "driving"
DATA_ROOT = Path("/datasets/driving/")

task_set.TASK_SET.clear()
task_set.TASK_SET.append(TASK_NAME)

import train_tokenizer
import train_dynamics

# Note for the future: 398x224 is a good resolution

def train_tok():
    args = argparse.Namespace(
        data_dirs=[str(DATA_ROOT)],
        ckpt_dir=str(DATA_ROOT / "tokenizer_ckpts"),
        batch_size=4,
        grad_accum=64,
        grad_clip=1.0,
        max_steps=2_000_000, # Would take ~9 days
        use_loss_norm=True,
        lr=2.5e-4,
        lr_scheduler="cosine",
        lr_warmup_steps=40_000,
        lr_min=1e-6,
        weight_decay=0.075,
        qk_norm=True,
        soft_cap=False,     # This does nothing
        save_every=100,     # This is every 100 optimizer steps, not normal steps
        seq_len=8,
        num_workers=10,
        H=400, W=224, C=3, patch=16,
        d_model=768, n_heads=12, depth=12, n_latents=256, d_bottleneck=64,
        dropout=0.05, mlp_ratio=4.0, time_every=4,
        mae_p_min=0.0, mae_p_max=0.85,
        wandb_project="dreamer4",
        wandb_run_name="tokenizer_robotics_RoPE_QKNorm",
        wandb_entity=None,
        log_every=100, print_every=100, viz_every=2000,
        viz_max_items=3, viz_max_T=4,
        lpips_weight=0.2,
        lpips_net="alex", lpips_frac=0.5,
        resume=None, seed=0, compile=True,
        grad_checkpointing=True,
    )
    train_tokenizer.train(args)

def train_dyn():
    pass
    # print("[Wrapper] Dynamics: Depth 12 + Bootstrapping")
    # tokenizer_ckpt = LOG_DIR / "tokenizer_ckpts" / "latest.pt"
    # args = argparse.Namespace(
    #     data_dirs=[str("")],
    #     frame_dirs=[str("")],
    #     tasks_json=str(""),
    #     tokenizer_ckpt=str(tokenizer_ckpt),
    #     ckpt_dir=str(LOG_DIR / "dynamics_ckpts"),
    #     batch_size=2,
    #     grad_accum=8,
    #     max_steps=15000,
    #     lr=2e-4,
    #     lr_scheduler="cosine",
    #     lr_warmup_steps=1500,
    #     lr_min=1e-6,
    #     weight_decay=0.01,
    #     grad_clip=1.0,
    #     save_every=2000,
    #     seq_len=16,
    #     num_workers=4,
    #     H=64, W=64, C=3, patch=8,
    #     d_model_dyn=512, dyn_depth=12, n_heads=8,
    #     dropout=0.1, mlp_ratio=4.0, time_every=1,
    #     packing_factor=2, n_register=4, n_agent=0,
    #     space_mode="wm_agent_isolated",
    #     k_max=8,
    #     bootstrap_start=2000,
    #     self_fraction=0.2,    # Crucial for long-term stability
    #     use_actions=True,
    #     wandb_project="dreamer4-pro",
    #     wandb_run_name="dynamics-pro",
    #     wandb_entity=None,
    #     log_every=100, eval_every=1000,
    #     eval_batch_size=2, eval_max_items=2, eval_ctx=4, eval_horizon=12,
    #     eval_schedule="shortcut", eval_d=0.25,
    #     resume=None, seed=0, compile=False
    # )
    # train_dynamics.train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tokenizer", "dynamics"])
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    if args.stage == "tokenizer": train_tok()
    else: raise RuntimeError("There are several issues with dynamics. Changes in train_tokenizer + hyperparms have to be ported over.")
