import os
import bisect
from pathlib import Path
from typing import Sequence, List, Dict, Union, Optional

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


class Mp4FrameDataset(Dataset):
    """
    Samples contiguous sequences directly from compressed MP4s.
    Returns: (T, 3, H, W) uint8 RGB tensor, decoded directly into VRAM.
    """

    def __init__(
            self,
            dir: Sequence[str | Path] | str | Path,
            seq_len: int = 12,
            iid_sampling: bool = True,
    ):
        super().__init__()
        assert dir is not None, "dir must be specified"

        if isinstance(dir, (list | tuple)):
            self.dir = Path(dir[0])
            if len(dir) > 1: raise RuntimeError("no too long dont want to implement support for this. just simlink everything cause i am bum")

        if isinstance(dir, str):
            self.dir = Path(dir)
        else:
            self.dir = dir

        self.seq_len = int(seq_len)
        self.iid_sampling = bool(iid_sampling)

        self.shards: List[Dict] = []
        self.cum_starts: List[int] = []
        total_starts = 0

        for fname in sorted(os.listdir(dir)):
            if not fname.endswith(".mp4"):
                continue
            path = dir / fname

            try:
                # Probe metadata on CPU to avoid allocating VRAM during initialization
                probe = VideoDecoder(str(path), device="cpu")
                N = len(probe)
            except Exception as e:
                print(f"[Mp4FrameDataset] Skipping {path} (load error): {e}")
                continue

            if N < self.seq_len:
                print(f"[Mp4FrameDataset] Skipping {path} (N={N} < seq_len={self.seq_len})")
                continue

            num_starts = N - self.seq_len + 1
            self.shards.append({"path": str(path), "num_frames": N, "num_starts": num_starts})
            total_starts += num_starts
            self.cum_starts.append(total_starts)

        self.total_starts = total_starts
        if self.total_starts > 0:
            print(f"[Mp4FrameDataset] shards={len(self.shards):,}, seq_starts={self.total_starts:,}")

        # Per-worker cache to avoid re-initializing the decoder for contiguous reads
        self._cache_path: Optional[str] = None
        self._cache_decoder: Optional[VideoDecoder] = None

    def __len__(self) -> int:
        return self.total_starts

    def _get_decoder(self, path: str) -> VideoDecoder:
        if self._cache_path == path and self._cache_decoder is not None:
            return self._cache_decoder

        # Initialize decoder on the target GPU. NVDEC handles the heavy lifting here.
        decoder = VideoDecoder(path, num_ffmpeg_threads=os.cpu_count() // 2)
        self._cache_path = path
        self._cache_decoder = decoder
        return decoder

    def _map_global_start_to_shard(self, global_start: int) -> tuple[int, int]:
        shard_idx = bisect.bisect_right(self.cum_starts, global_start)
        prev_cum = 0 if shard_idx == 0 else self.cum_starts[shard_idx - 1]
        return shard_idx, global_start - prev_cum

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.total_starts == 0:
            raise IndexError("Empty dataset")

        if self.iid_sampling:
            global_start = torch.randint(0, self.total_starts, (1,)).item()
        else:
            global_start = int(idx)

        shard_idx, start = self._map_global_start_to_shard(global_start)
        meta = self.shards[shard_idx]

        decoder = self._get_decoder(meta["path"])
        end = start + self.seq_len

        # Slicing the decoder directly yields a (T, 3, H, W) RGB tensor in uint8
        seq_u8 = decoder[start:end]
        return seq_u8

