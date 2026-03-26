# model.py
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Modality(IntEnum):
    LATENT = -1
    IMAGE = 0
    ACTION = 1
    PROPRIO = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7


@dataclass(frozen=True)
class TokenLayout:
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        parts =[]
        if self.n_latents > 0:
            parts.append(torch.full((self.n_latents,), int(Modality.LATENT), dtype=torch.int32))
        for m, n in self.segments:
            if n > 0:
                parts.append(torch.full((n,), int(m), dtype=torch.int32))
        return torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        idx = 0
        out: Dict[Modality, slice] = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out


def temporal_patchify(videos_btchw: torch.Tensor, patch: int) -> torch.Tensor:
    B, T, C, H, W = videos_btchw.shape
    # View as (B, T, C, GridH, PatchH, GridW, PatchW)
    x = videos_btchw.view(B, T, C, H // patch, patch, W // patch, patch)
    # Permute to (B, T, GridH, GridW, C, PatchH, PatchW)
    x = x.permute(0, 1, 3, 5, 2, 4, 6)
    # Flatten to (B, T, Np, Dp)
    x = x.reshape(B, T, -1, C * patch * patch)
    return x

def temporal_unpatchify(patches_btnd: torch.Tensor, H: int, W: int, C: int, patch: int) -> torch.Tensor:
    B, T, Np, Dp = patches_btnd.shape
    # View as (B, T, GridH, GridW, C, PatchH, PatchW)
    x = patches_btnd.view(B, T, H // patch, W // patch, C, patch, patch)
    # Permute to (B, T, C, GridH, PatchH, GridW, PatchW)
    x = x.permute(0, 1, 4, 2, 5, 3, 6)
    # Flatten to (B, T, C, H, W)
    x = x.reshape(B, T, C, H, W)
    return x


def sinusoid_table(n: int, d: int, base: float = 10000.0, device=None) -> torch.Tensor:
    # fp32 by construction
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)  # (n,1)
    i   = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)  # (1,d)
    k   = torch.floor(i / 2.0)
    # stable: exp(log(base) * exponent)
    div = torch.exp(-(2.0 * k) / max(1.0, float(d)) * math.log(base))
    ang = pos * div
    return torch.where((i % 2) == 0, torch.sin(ang), torch.cos(ang))  # (n,d) fp32


def add_sinusoidal_positions(tokens_btSd: torch.Tensor) -> torch.Tensor:
    B, T, S, D = tokens_btSd.shape
    device = tokens_btSd.device
    pos_t = sinusoid_table(T, D, device=device)  # fp32
    pos_s = sinusoid_table(S, D, device=device)  # fp32
    pos = (pos_t[None, :, None, :] + pos_s[None, None, :, :])
    return tokens_btSd + pos.to(dtype=tokens_btSd.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to Q or K tensors. x: (B, L, H, D)"""
    d = x.shape[-1]
    # Split features into pairs
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)

    # Broadcast cos and sin to match x's shape if needed
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, L, 1, D)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, L, 1, D)

    return x * cos + rotated * sin


def get_1d_rope_freqs(seq_len: int, dim: int, device: torch.device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
    return freqs.cos(), freqs.sin()


def get_2d_rope_freqs(h: int, w: int, dim: int, device: torch.device, base: float = 10000.0):
    # We need half the total dimension for the 2D grid
    half_dim = dim // 2

    # Calculate base frequencies for 1/4th of the dimension
    quart_dim = half_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, quart_dim * 2, 2, device=device).float() / half_dim))

    t_y = torch.arange(h, device=device).float()
    t_x = torch.arange(w, device=device).float()

    freqs_y = torch.outer(t_y, inv_freq)  # (h, quart_dim)
    freqs_x = torch.outer(t_x, inv_freq)  # (w, quart_dim)

    # Broadcast to grid
    freqs_y = freqs_y.view(h, 1, -1).expand(h, w, -1)
    freqs_x = freqs_x.view(1, w, -1).expand(h, w, -1)

    # Concat Y and X frequencies -> (h, w, half_dim)
    freqs_2d = torch.cat([freqs_y, freqs_x], dim=-1)

    # Duplicate to match apply_rope's [-x2, x1] structure -> (h, w, dim)
    freqs_2d = torch.cat([freqs_2d, freqs_2d], dim=-1).reshape(h * w, dim)

    return freqs_2d.cos(), freqs_2d.sin()


class MAEReplacer(nn.Module):
    def __init__(self, d_model: int, p_min: float = 0.0, p_max: float = 0.9):
        super().__init__()
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.mask_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patches_btnd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patches: (B,T,Np,D)
        returns:
          replaced: (B,T,Np,D)
          mae_mask: (B,T,Np,1) bool, True where masked (must reconstruct)
          keep_prob:(B,T,1) float
        """
        B, T, Np, D = patches_btnd.shape
        device = patches_btnd.device

        # fast path: deterministic "no MAE"
        if self.p_min == 0.0 and self.p_max == 0.0:
            keep_prob = torch.ones((B, T, 1), device=device, dtype=patches_btnd.dtype)
            mae_mask = torch.zeros((B, T, Np, 1), device=device, dtype=torch.bool)
            return patches_btnd, mae_mask, keep_prob

        p_bt = torch.empty((B, T), device=device).uniform_(self.p_min, self.p_max)
        keep_prob = (1.0 - p_bt).unsqueeze(-1)                          # (B,T,1)
        keep = (torch.rand((B, T, Np), device=device) < keep_prob)      # (B,T,Np)
        keep_ = keep.unsqueeze(-1)
        mask_tok = self.mask_token.to(dtype=patches_btnd.dtype)
        replaced = torch.where(keep_, patches_btnd, mask_tok.view(1, 1, 1, D))
        mae_mask = (~keep_).to(torch.bool)
        return replaced, mae_mask, keep_prob


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (self.scale / torch.sqrt(var + self.eps))


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc_in = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        h = u * F.silu(v)
        h = self.drop(h)
        y = self.fc_out(h)
        y = self.drop(y)
        return y


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, qk_norm: bool = False, soft_cap: float = 0.0):
        super().__init__()
        # assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)
        self.qk_norm = qk_norm
        self.soft_cap = soft_cap

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x_nld:
    torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False, rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None):
        N, L, D = x_nld.shape
        q, k, v = self.qkv(x_nld).chunk(3, dim=-1)

        # Reshape to (Batch, Seq_len, Heads, Head_dim)
        q = q.view(N, L, self.n_heads, self.head_dim)
        k = k.view(N, L, self.n_heads, self.head_dim)
        v = v.view(N, L, self.n_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        # Transpose for SDPA: (Batch, Heads, Seq_len, Head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        drop = self.dropout_p if self.training else 0.0

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=is_causal)
            
        y = y.transpose(1, 2).contiguous().view(N, L, D)
        return self.out(y)


class SpaceSelfAttentionModality(nn.Module):
    def __init__(self, d_model: int, n_heads: int, modality_ids: torch.Tensor, n_latents: int, mode: str, dropout: float, qk_norm: bool = False, soft_cap: float = 0.0):
        super().__init__()
        self.n_latents = int(n_latents)
        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)
        device = modality_ids.device

        S = int(self.modality_ids.numel())
        allow = self._build_allow(S)
        attn_mask = allow.unsqueeze(0).unsqueeze(0)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout, qk_norm=qk_norm, soft_cap=soft_cap)

        spatial_mask = (self.modality_ids == int(Modality.IMAGE)) | (self.modality_ids == int(Modality.SPATIAL))
        num_spatial = spatial_mask.sum().item()

        rope_cos = torch.ones((S, self.attn.head_dim), dtype=torch.float32, device=device)
        rope_sin = torch.zeros((S, self.attn.head_dim), dtype=torch.float32, device=device)

        if num_spatial > 0:
            grid_size = int(math.sqrt(num_spatial))
            assert grid_size * grid_size == num_spatial, "Spatial tokens must form a square grid for 2D RoPE"

            cos_2d, sin_2d = get_2d_rope_freqs(grid_size, grid_size, self.attn.head_dim, device=device)

            rope_cos[spatial_mask] = cos_2d
            rope_sin[spatial_mask] = sin_2d

        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def _build_allow(self, S: int) -> torch.Tensor:
        device = self.modality_ids.device
        q_idx = torch.arange(S, device=device).unsqueeze(1)  # (S,1)
        k_idx = torch.arange(S, device=device).unsqueeze(0)  # (1,S)

        is_q_lat = q_idx < self.n_latents
        is_k_lat = k_idx < self.n_latents

        q_mod = self.modality_ids[q_idx]
        k_mod = self.modality_ids[k_idx]
        same_mod = (q_mod == k_mod)

        if self.mode == "encoder":
            allow_lat_q = torch.ones((S, S), dtype=torch.bool, device=device)
            allow_nonlat_q = same_mod
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "decoder":
            allow_lat_q = is_k_lat
            allow_nonlat_q = same_mod | is_k_lat
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "wm_agent":
            # full mixing across modalities
            return torch.ones((S, S), dtype=torch.bool, device=device)

        if self.mode == "wm_agent_isolated":
            # non-agent tokens: can attend to everything EXCEPT agent tokens
            # agent tokens: attend only to agent tokens (keeps them inert in pretrain)
            is_q_agent = (q_mod == int(Modality.AGENT))
            is_k_agent = (k_mod == int(Modality.AGENT))

            allow = torch.ones((S, S), dtype=torch.bool, device=device)

            # non-agent queries cannot see agent keys
            allow_non_agent_q = ~is_q_agent
            allow = torch.where(allow_non_agent_q, ~is_k_agent, allow)

            # agent queries only see agent keys
            allow = torch.where(is_q_agent, is_k_agent, allow)
            return allow

        raise ValueError(f"Unsupported mode for tokenizer/wm: {self.mode}")

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        x = x_btSd.reshape(B * T, S, D)

        mask = self.attn_mask.expand(B * T, 1, S, S)

        y = self.attn(x, attn_mask=mask, is_causal=False, rope_cos=self.rope_cos, rope_sin=self.rope_sin)

        return y.reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, latents_only: bool, n_latents: int, device: torch.device, max_time_steps: int = 4096, qk_norm: bool = False, soft_cap: float = 0.0):
        super().__init__()
        self.latents_only = bool(latents_only)
        self.n_latents = int(n_latents)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout, qk_norm=qk_norm, soft_cap=soft_cap)

        cos_1d, sin_1d = get_1d_rope_freqs(max_time_steps, self.attn.head_dim, device=device)
        self.register_buffer("rope_cos", cos_1d, persistent=False)
        self.register_buffer("rope_sin", sin_1d, persistent=False)

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape

        assert T <= self.rope_cos.shape[0], f"Sequence length {T} exceeds cached maximum of {self.rope_cos.shape[0]}"

        rope_cos = self.rope_cos[:T]
        rope_sin = self.rope_sin[:T]

        if self.latents_only:
            L = self.n_latents
            lat = x_btSd[:, :, :L, :]
            lat_nld = lat.permute(0, 2, 1, 3).contiguous().view(B * L, T, D)

            out = self.attn(lat_nld, is_causal=True, rope_cos=rope_cos, rope_sin=rope_sin)

            out = out.view(B, L, T, D).permute(0, 2, 1, 3).contiguous()
            x = x_btSd.clone()
            x[:, :, :L, :] = out
            return x
        else:
            x_nld = x_btSd.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
            out = self.attn(x_nld, is_causal=True, rope_cos=rope_cos, rope_sin=rope_sin)
            return out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()


class BlockCausalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        layer_index: int,
        time_every: int,
        latents_only_time: bool,
        qk_norm: bool = False,
        soft_cap: float = 0.0,
    ):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)
        device = modality_ids.device

        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttentionModality(d_model, n_heads, modality_ids, n_latents, space_mode, dropout, qk_norm=qk_norm, soft_cap=soft_cap)
        self.drop1 = nn.Dropout(dropout)

        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model, n_heads, dropout, latents_only_time, n_latents, device=device, qk_norm=qk_norm, soft_cap=soft_cap)
            self.drop2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.space(self.norm1(x)))
        if self.do_time:
            # new_x = x_unromed + x_normed_with_new_latents
            x = x + self.drop2(self.time(self.norm2(x)))
        x = x + self.mlp(self.norm3(x))
        return x


class BlockCausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        time_every: int,
        latents_only_time: bool,
        gradient_checkpointing: bool = False,
        qk_norm: bool = False,
        soft_cap: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BlockCausalLayer(
                d_model=d_model, n_heads=n_heads, n_latents=n_latents,
                modality_ids=modality_ids, space_mode=space_mode,
                dropout=dropout, mlp_ratio=mlp_ratio,
                layer_index=i, time_every=time_every,
                latents_only_time=latents_only_time,
                qk_norm=qk_norm, soft_cap=soft_cap,
            )
            for i in range(depth)
        ])

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        patch_dim: int,
        d_model: int,
        n_latents: int,
        n_patches: int,
        n_heads: int,
        depth: int,
        d_bottleneck: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        mae_p_min: float = 0.0,
        mae_p_max: float = 0.9,
        gradient_checkpointing: bool = False,
        qk_norm: bool = False,
        soft_cap: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_patches = n_patches

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)

        self.layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        modality_ids = self.layout.modality_ids().to(self.patch_proj.weight.device)

        self.transformer = BlockCausalTransformer(
            d_model=d_model, n_heads=n_heads, depth=depth,
            n_latents=n_latents, modality_ids=modality_ids,
            space_mode="encoder",
            dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=time_every, latents_only_time=latents_only_time,
            gradient_checkpointing=gradient_checkpointing,
            qk_norm=qk_norm, soft_cap=soft_cap
        )
        self.mae = MAEReplacer(d_model=d_model, p_min=mae_p_min, p_max=mae_p_max)

        self.latents = nn.Parameter(torch.empty(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)

    def forward(self, patch_tokens_btnd: torch.Tensor):
        B, T, Np, Dp = patch_tokens_btnd.shape
        # assert Np == self.n_patches

        proj = self.patch_proj(patch_tokens_btnd)            # (B,T,Np,D)
        proj_masked, mae_mask, keep_prob = self.mae(proj)    # (B,T,Np,D), (B,T,Np,1), (B,T,1)

        lat = self.latents.view(1, 1, self.n_latents, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, proj_masked], dim=2)        # (B,T,S,D)
        # tokens = add_sinusoidal_positions(tokens)

        enc = self.transformer(tokens)
        z = torch.tanh(self.bottleneck_proj(enc[:, :, :self.n_latents, :]))
        return z, (mae_mask, keep_prob)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        d_bottleneck: int,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        n_patches: int,
        d_patch: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        gradient_checkpointing: bool = False,
        qk_norm: bool = False,
        soft_cap: float = 0.0
    ):
        super().__init__()
        self.n_latents = n_latents
        self.n_patches = n_patches

        self.up_proj = nn.Linear(d_bottleneck, d_model)
        self.patch_queries = nn.Parameter(torch.empty(n_patches, d_model))
        nn.init.normal_(self.patch_queries, std=0.02)

        self.patch_head = nn.Linear(d_model, d_patch)

        self.layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        modality_ids = self.layout.modality_ids().to(self.up_proj.weight.device)

        self.transformer = BlockCausalTransformer(
            d_model=d_model, n_heads=n_heads, depth=depth,
            n_latents=n_latents, modality_ids=modality_ids,
            space_mode="decoder",
            dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=time_every, latents_only_time=latents_only_time,
            gradient_checkpointing=gradient_checkpointing,
            qk_norm=qk_norm, soft_cap=soft_cap
        )

    def forward(self, z_btLd: torch.Tensor) -> torch.Tensor:
        B, T, L, _ = z_btLd.shape
        # assert L == self.n_latents

        lat = torch.tanh(self.up_proj(z_btLd))                                 # (B,T,L,D)
        qry = self.patch_queries.view(1, 1, self.n_patches, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, qry], dim=2)                                  # (B,T,S,D)
        # tokens = add_sinusoidal_positions(tokens)

        x = self.transformer(tokens)
        x_p = x[:, :, L:, :]
        return torch.sigmoid(self.patch_head(x_p))                             # (B,T,Np,Dp)


class Tokenizer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, patches_btnd: torch.Tensor):
        z, (mae_mask, keep_prob) = self.encoder(patches_btnd)
        pred = self.decoder(z)
        return pred, mae_mask, keep_prob


def pack_bottleneck_to_spatial(z_btLd: torch.Tensor, *, n_spatial: int, k: int) -> torch.Tensor:
    """
    z: (B,T,L,D_b) where L == n_spatial * k
    -> (B,T,n_spatial,D_b*k)
    """
    B, T, L, D = z_btLd.shape
    # assert L == n_spatial * k, f"L={L} must equal n_spatial*k={n_spatial*k}"
    return z_btLd.view(B, T, n_spatial, k * D)


def unpack_spatial_to_bottleneck(z_btSd: torch.Tensor, *, k: int) -> torch.Tensor:
    """
    z: (B,T,n_spatial,D_b*k) -> (B,T,n_spatial*k,D_b)
    """
    B, T, S, DK = z_btSd.shape
    # assert DK % k == 0, f"D={DK} must be divisible by k={k}"
    D = DK // k
    return z_btSd.view(B, T, S * k, D)


class ActionEncoder(nn.Module):
    """
    Continuous actions in [-1,1], shape (B,T,A) -> token (B,T,1,D).
    If actions is None (unlabeled pretrain), emits a learned base token.
    """
    def __init__(self, d_model: int, action_dim: int = 16, hidden_mult: float = 2.0):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)

        hidden = int(self.d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.base, std=0.02)

        self.fc1 = nn.Linear(self.action_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.d_model)

        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],                 # (B,T,A) or None
        *,
        batch_time_shape: Optional[Tuple[int,int]] = None,
        act_mask: Optional[torch.Tensor] = None,         # (B,T,A) or (A,)
        as_tokens: bool = True,
    ) -> torch.Tensor:
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            x = actions
            if act_mask is not None:
                x = x * act_mask
            x = x.clamp(-1, 1)
            out = self.fc2(F.silu(self.fc1(x))) + self.base.view(1, 1, -1)

        return out[:, :, None, :] if as_tokens else out


class TaskEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_agent: int = 1,
        use_ids: bool = True,
        n_tasks: int = 128,
        d_task: int = 64,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_agent = int(n_agent)
        self.use_ids = bool(use_ids)
        self.n_tasks = int(n_tasks)
        self.d_task = int(d_task)

        if self.use_ids:
            self.task_table = nn.Embedding(self.n_tasks, self.d_model)
        else:
            self.task_proj = nn.Linear(self.d_task, self.d_model)

        self.agent_base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.agent_base, std=0.02)

    def forward(self, task: torch.Tensor, *, B: int, T: int) -> torch.Tensor:
        if self.use_ids:
            emb = self.task_table(task.to(torch.long))  # (B,D)
        else:
            emb = self.task_proj(task)                  # (B,D)

        x = emb + self.agent_base.view(1, -1)          # (B,D)
        return x[:, None, None, :].expand(B, T, self.n_agent, self.d_model)


class Dynamics(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_bottleneck: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int,
        n_agent: int,
        n_heads: int,
        depth: int,
        k_max: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        space_mode: str = "wm_agent_isolated",  # or "wm_agent"
        qk_norm: bool = False,
        soft_cap: float = 0.0
    ):
        super().__init__()
        assert d_spatial % d_bottleneck == 0, "expected packing: d_spatial = d_bottleneck * packing_factor"
        self.d_model = int(d_model)
        self.d_spatial = int(d_spatial)
        self.n_spatial = int(n_spatial)
        self.n_register = int(n_register)
        self.n_agent = int(n_agent)
        self.k_max = int(k_max)

        self.spatial_proj = nn.Linear(self.d_spatial, self.d_model)
        self.register_tokens = nn.Parameter(torch.empty(self.n_register, self.d_model))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.action_encoder = ActionEncoder(d_model=self.d_model, action_dim=16)

        # shortcut conditioning
        self.num_step_bins = int(math.log2(self.k_max)) + 1
        self.step_embed = nn.Embedding(self.num_step_bins, self.d_model)
        self.signal_embed = nn.Embedding(self.k_max + 1, self.d_model)

        segments =[
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),
            (Modality.SHORTCUT_STEP, 1),
            (Modality.SPATIAL, self.n_spatial),
            (Modality.REGISTER, self.n_register),
        ]
        if self.n_agent > 0:
            segments.append((Modality.AGENT, self.n_agent))

        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = self.layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.agent_slice = sl.get(Modality.AGENT, slice(0, 0))
        modality_ids = self.layout.modality_ids().to(self.spatial_proj.weight.device)

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=int(n_heads),
            depth=int(depth),
            n_latents=0,
            modality_ids=modality_ids,
            space_mode=space_mode,
            dropout=float(dropout),
            mlp_ratio=float(mlp_ratio),
            time_every=int(time_every),
            latents_only_time=False,
            qk_norm=qk_norm,
            soft_cap=soft_cap
        )

        self.flow_x_head = nn.Linear(self.d_model, self.d_spatial)
        nn.init.zeros_(self.flow_x_head.weight)
        nn.init.zeros_(self.flow_x_head.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],          # (B,T,16) or None
        step_idxs: torch.Tensor,                  # (B,T)
        signal_idxs: torch.Tensor,                # (B,T)
        packed_enc_tokens: torch.Tensor,          # (B,T,n_spatial,d_spatial)
        *,
        act_mask: Optional[torch.Tensor] = None,  # (B,T,16) or (16,) or None
        agent_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = packed_enc_tokens.shape[:2]

        spatial_tokens = self.spatial_proj(packed_enc_tokens)  # (B,T,n_spatial,d_model)

        action_tokens = self.action_encoder(
            actions,
            batch_time_shape=(B, T),
            act_mask=act_mask,
            as_tokens=True,
        )  # (B,T,1,d_model)

        reg = self.register_tokens.view(1, 1, self.n_register, self.d_model).expand(B, T, -1, -1)

        step_tok = self.step_embed(step_idxs.to(torch.long))[:, :, None, :]
        sig_tok = self.signal_embed(signal_idxs.to(torch.long))[:, :, None, :]

        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros((B, T, self.n_agent, self.d_model), device=spatial_tokens.device, dtype=spatial_tokens.dtype)
            toks =[action_tokens, sig_tok, step_tok, spatial_tokens, reg, agent_tokens]
        else:
            toks =[action_tokens, sig_tok, step_tok, spatial_tokens, reg]

        tokens = torch.cat(toks, dim=2)  # (B,T,S,D)
        # tokens = add_sinusoidal_positions(tokens)
        x = self.transformer(tokens)

        spatial_out = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_out)  # (B,T,n_spatial,d_spatial)

        h_t = None
        if self.n_agent > 0:
            h_t = x[:, :, self.agent_slice, :]   # (B,T,n_agent,d_model)

        return x1_hat, h_t


def recon_loss_from_mae(pred_btnd: torch.Tensor,
                        target_btnd: torch.Tensor,
                        mae_mask_btNp1: torch.Tensor) -> torch.Tensor:
    # mask: (B,T,Np,1) bool, True where masked
    mask = mae_mask_btNp1.to(dtype=torch.float32)  # (B,T,Np,1)

    # compute in fp32 to avoid fp16 overflow on reduction
    diff = (pred_btnd.float() - target_btnd.float())          # (B,T,Np,Dp)
    sq = diff.mul(diff) * mask                                # broadcast mask over Dp
    denom = mask.sum().clamp_min(1.0) * diff.shape[-1]        # (#masked patches) * Dp
    return sq.sum() / denom


def lpips_on_mae_recon(
    lpips_fn,
    pred_btnd: torch.Tensor,
    target_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
    *,
    H: int, W: int, C: int, patch: int,
    subsample_frac: float = 1.0,
) -> torch.Tensor:
    recon_masked_btnd = torch.where(mae_mask_btNp1, pred_btnd, target_btnd)
    recon = temporal_unpatchify(recon_masked_btnd.float(), H, W, C, patch)
    tgt   = temporal_unpatchify(target_btnd.float(),        H, W, C, patch)

    if subsample_frac < 1.0:
        B, T = recon.shape[:2]
        step = max(1, int(1.0 / subsample_frac))
        recon = recon[:, ::step]
        tgt   = tgt[:, ::step]

    recon = (recon.clamp(0, 1) * 2.0 - 1.0).float()
    tgt   = (tgt.clamp(0, 1)   * 2.0 - 1.0).float()

    B, T = recon.shape[:2]
    recon = recon.reshape(B * T, C, H, W)
    tgt   = tgt.reshape(B * T, C, H, W)

    with torch.autocast(device_type="cuda", enabled=False):
        lp = lpips_fn(recon, tgt)
    return lp.mean()
