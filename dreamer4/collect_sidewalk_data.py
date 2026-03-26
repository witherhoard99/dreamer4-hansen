import os
import glob
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# --- CONFIGURATION ---
INPUT_ROOT = Path("/datasets/sidewalk/")            # This directory needs to be exist
OUTPUT_ROOT = Path("/datasets/processed/sidewalk/")  # This directory needs to be made
TASK_NAME = "robot_sidewalk"
SHARD_SIZE = 2048
TARGET_SIZE = (384, 384)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # Process 32 images at once on GPU
NUM_WORKERS = max(1, os.cpu_count() - 2)

RAW_DIR = OUTPUT_ROOT / "raw"
SHARD_DIR = OUTPUT_ROOT / "shards" / TASK_NAME


class RawImageDataset(Dataset):
    """
    Helper class to load images in parallel using DataLoader workers.
    Returns the raw tensor and the episode ID.
    """

    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, episode_id = self.file_list[idx]
        with Image.open(path).convert("RGB") as img:
            # Convert to tensor (C, H, W) float32 [0,1]
            # We do NOT resize here; we let the GPU do that in batches.
            return F.to_tensor(img), episode_id


def process_robot_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Scrape all paths first (Metadata collection)
    traj_folders = sorted(glob.glob(str(INPUT_ROOT / "traj_nav_*")))
    print(f"Found {len(traj_folders)} trajectory folders.")

    # We flatten the structure first to feed into a DataLoader
    # List of tuples: (image_path, episode_id)
    all_files_meta = []

    print("Indexing files...")
    for episode_id, folder in enumerate(traj_folders):
        img_paths = sorted(
            glob.glob(os.path.join(folder, "*.png")) +
            glob.glob(os.path.join(folder, "*.jpg"))
        )
        for p in img_paths:
            all_files_meta.append((p, episode_id))

    print(f"Total frames found: {len(all_files_meta)}")
    print(f"Processing with {NUM_WORKERS} CPU workers on {DEVICE}")

    # 2. Setup Parallel Loader
    dataset = RawImageDataset(all_files_meta)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True  # Faster transfer to CUDA
    )

    all_frames_buffer = []
    all_episodes_buffer = []
    shard_count = 0

    def save_shard(frames_list, index):
        path = SHARD_DIR / f"{TASK_NAME}_shard{index:04d}.pt"
        tensor = torch.stack(frames_list)  # Already uint8 (N, C, H, W)
        torch.save({"frames": tensor}, path)
        print(f" Saved shard {index} to {path}")

    # 3. Iterate through batches
    for batch_imgs, batch_episodes in tqdm(loader, desc="Processing Batches"):
        # Move batch to GPU
        # shape: (Batch, C, H, W_raw)
        batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)

        # Batch Resize on GPU (Much faster than single image resize)
        # Result shape: (Batch, C, 384, 384)
        batch_resized = F.resize(
            batch_imgs,
            TARGET_SIZE,
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True
        )

        # Convert to uint8
        batch_uint8 = (batch_resized * 255).byte()

        # Move back to CPU memory to accumulate for the shard
        # (keeping 2048 images in VRAM might crash it)
        # If you have massive VRAM, you can keep them there, but safer to move back.
        batch_uint8 = batch_uint8.cpu()

        # Unbind batch into individual items to append to buffer
        frames = torch.unbind(batch_uint8, dim=0)

        all_frames_buffer.extend(frames)
        all_episodes_buffer.extend(batch_episodes.tolist())

        # Save shard if buffer is full
        while len(all_frames_buffer) >= SHARD_SIZE:
            # Slice off the first SHARD_SIZE
            chunk = all_frames_buffer[:SHARD_SIZE]
            save_shard(chunk, shard_count)
            shard_count += 1

            # Remove from buffer
            all_frames_buffer = all_frames_buffer[SHARD_SIZE:]

    # Save remaining frames
    if all_frames_buffer:
        save_shard(all_frames_buffer, shard_count)

    # 4. Create the "Demo" file for Dynamics training
    print("Generating dynamics metadata (raw/nav_robot.pt)...")
    total_n = len(all_files_meta)

    demo_data = {
        "episode": torch.tensor(all_episodes_buffer, dtype=torch.long),  # Saved on CPU to save VRAM
        "action": torch.zeros((total_n, 16), dtype=torch.float32),
        "reward": torch.zeros((total_n,), dtype=torch.float32),
    }
    torch.save(demo_data, RAW_DIR / f"{TASK_NAME}.pt")

    # 5. Create tasks.json
    tasks_meta = {
        TASK_NAME: {
            "action_dim": 1,
            "text_embedding": [0.0] * 512
        }
    }
    with open(OUTPUT_ROOT / "tasks.json", "w") as f:
        json.dump(tasks_meta, f)

    print("\nPre-processing complete!")
    print(f"Shards saved to: {SHARD_DIR}")
    print(f"Dynamics file saved to: {RAW_DIR / f'{TASK_NAME}.pt'}")


if __name__ == "__main__":
    # Windows requires this guard for multiprocessing
    process_robot_data()
