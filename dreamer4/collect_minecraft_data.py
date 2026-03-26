import os
import glob
import json
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
TASK_NAME = "minecraft"
SOURCE_DIR = "/datasets/minecraft/128x128_minecraft/"  # Your source folder
SHARD_SIZE = 16384  # Frames per shard file
IMAGE_SIZE = (128, 128)  # Target size (H, W)

# Paths setup (Mirroring train_wrapper structure)
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_ROOT = SCRIPT_DIR / "data"
RAW_DIR = DATA_ROOT / "raw"
SHARDS_DIR = DATA_ROOT / "shards"
TASKS_JSON = DATA_ROOT / "tasks.json"


def save_shard(frames, task_shard_dir, shard_count):
    """Saves a chunk of frames to a .pt file"""
    shard_path = task_shard_dir / f"{TASK_NAME}_shard{shard_count:04d}.pt"

    # Ensure strict format: (N, 3, H, W) uint8
    if frames.dtype != torch.uint8:
        frames = frames.to(torch.uint8)

    torch.save({"frames": frames.clone()}, shard_path)
    return shard_path


def update_tasks_json():
    """Updates tasks.json with metadata for the new task"""
    if TASKS_JSON.exists():
        with open(TASKS_JSON, 'r') as f:
            meta = json.load(f)
    else:
        meta = {}

    # Minecraft is usually unlabeled video, so we define dummy action dims
    # This allows the Dynamics model to run in "unconditional" mode effectively
    meta[TASK_NAME] = {
        "action_dim": 16,  # Default latent action dim
        "domain": "video",
        "description": "Minecraft 128x128 video dataset"
    }

    with open(TASKS_JSON, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[Metadata] Updated {TASKS_JSON}")


def main():
    # 1. Setup Directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    task_shard_dir = SHARDS_DIR / TASK_NAME
    task_shard_dir.mkdir(parents=True, exist_ok=True)

    # 2. Find Videos
    video_paths = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.mp4")))
    if not video_paths:
        print(f"[Error] No .mp4 files found in {SOURCE_DIR}")
        return

    print(f"[Convert] Found {len(video_paths)} videos. Processing...")

    # Buffers
    frames_buffer = []

    # Metadata Buffers (needed for Dynamics training, even if dummy)
    meta_episodes = []
    meta_actions = []
    meta_rewards = []

    shard_count = 0
    total_frames = 0
    current_episode_id = 0

    for vid_path in tqdm(video_paths):
        try:
            # Read Video: (T, H, W, C) uint8 [0-255]
            # using torchvision is standard, requires ffmpeg installed
            vframes, _, _ = torchvision.io.read_video(vid_path, pts_unit='sec')

            if len(vframes) == 0:
                continue

            # Permute to (T, 3, H, W)
            vframes = vframes.permute(0, 3, 1, 2)

            # Resize if necessary (Center Crop or Resize)
            if (vframes.shape[2] != IMAGE_SIZE[0]) or (vframes.shape[3] != IMAGE_SIZE[1]):
                vframes = torchvision.transforms.functional.resize(vframes, IMAGE_SIZE)

            n_frames = len(vframes)

            # --- Metadata Generation ---
            # Create episode IDs for this video
            ep_ids = torch.full((n_frames,), current_episode_id, dtype=torch.long)

            # Create Dummy Actions (T, 16) - Zero vectors
            # This tells the model "no action was taken" or allows unguided learning
            actions = torch.zeros((n_frames, 16), dtype=torch.float32)

            # Create Dummy Rewards (T,)
            rewards = torch.zeros((n_frames,), dtype=torch.float32)

            # Append to buffers
            frames_buffer.append(vframes)

            # Append metadata to lists
            meta_episodes.append(ep_ids)
            meta_actions.append(actions)
            meta_rewards.append(rewards)

            total_frames += n_frames
            current_episode_id += 1

            # --- Shard Flushing ---
            # Check if we have enough frames to write a shard
            # We concatenate the current buffer to check size
            current_buffer_tensor = torch.cat(frames_buffer, dim=0)

            while len(current_buffer_tensor) >= SHARD_SIZE:
                # Slice off one shard
                to_save = current_buffer_tensor[:SHARD_SIZE]
                remainder = current_buffer_tensor[SHARD_SIZE:]

                save_shard(to_save, task_shard_dir, shard_count)
                shard_count += 1

                # Update buffer with remainder
                current_buffer_tensor = remainder
                frames_buffer = [remainder] if len(remainder) > 0 else []

        except Exception as e:
            print(f"Error processing {vid_path}: {e}")

    # Flush remaining frames
    if frames_buffer:
        final_tensor = torch.cat(frames_buffer, dim=0)
        if len(final_tensor) > 0:
            save_shard(final_tensor, task_shard_dir, shard_count)

    print(f"[Convert] Saved shards to {task_shard_dir}")

    # 3. Save Aggregate Metadata (Required for Dynamics Training)
    # The dynamics loader requires a raw .pt file to know episode boundaries
    print("[Convert] aggregating metadata...")
    if meta_episodes:
        full_ep = torch.cat(meta_episodes, dim=0)
        full_act = torch.cat(meta_actions, dim=0)
        full_rew = torch.cat(meta_rewards, dim=0)

        meta_path = RAW_DIR / f"{TASK_NAME}.pt"
        torch.save({
            "episode": full_ep,
            "action": full_act,
            "reward": full_rew
        }, meta_path)
        print(f"[Convert] Metadata saved to {meta_path} (Total frames: {len(full_ep)})")

    # 4. Update Tasks JSON
    update_tasks_json()


if __name__ == "__main__":
    main()
