import os
import sys
import json
import torch
from pathlib import Path

# --- CONFIG ---
TASK_NAME = "custom_sim"
IMAGE_SIZE = (64, 64)
EPISODES = 500
MAX_STEPS = 128  # Now strictly enforced

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_ROOT = SCRIPT_DIR / "data"
RAW_DIR = DATA_ROOT / "raw"
SHARDS_DIR = DATA_ROOT / "shards"
TASKS_JSON = DATA_ROOT / "tasks.json"

sys.path.append(str(SCRIPT_DIR))
from environment import Simulator


def save_data(frames, actions, rewards, episode_ids):
    all_frames = torch.stack(frames).cpu()
    if all_frames.dtype != torch.uint8:
        all_frames = (all_frames.clamp(0, 1) * 255).to(torch.uint8)

    task_shard_dir = SHARDS_DIR / TASK_NAME
    task_shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = task_shard_dir / f"{TASK_NAME}_shard0000.pt"
    torch.save({"frames": all_frames}, shard_path)

    meta_path = RAW_DIR / f"{TASK_NAME}.pt"
    torch.save({
        "episode": torch.tensor(episode_ids, dtype=torch.long),
        "action": torch.stack(actions).cpu().float(),
        "reward": torch.stack(rewards).cpu().float()
    }, meta_path)

    meta = {TASK_NAME: {"action_dim": 2, "domain": "sim"}}
    with open(TASKS_JSON, 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    env = Simulator(task=TASK_NAME, size=IMAGE_SIZE, save_video=False)
    frames_buf, act_buf, rew_buf, ep_buf = [], [], [], []

    print(f"[Collect] Starting collection of {EPISODES} episodes (Max {MAX_STEPS} steps each)...")
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        t = 0  # Initialize step counter

        while not done and t < MAX_STEPS:
            action = torch.rand(2).uniform_(-1, 1).to(env.device)

            # Record current observation and action taken
            frames_buf.append(obs['image'].cpu())
            ep_buf.append(ep)
            act_buf.append(action.cpu())

            # Step environment
            obs, reward, is_terminal, is_last, info = env.step(action)
            rew_buf.append(reward.cpu())

            done = is_terminal or is_last
            t += 1  # Increment step counter

        if (ep + 1) % 10 == 0:
            print(f"[Collect] Progress: {ep + 1}/{EPISODES} (Last ep steps: {t})")

    env.close()
    save_data(frames_buf, act_buf, rew_buf, ep_buf)
    print(f"[Collect] Data saved. Total frames: {len(frames_buf)}")


if __name__ == "__main__":
    main()