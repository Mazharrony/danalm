"""DanaLM: a small Llama-style decoder-only transformer (Phase 3, D-026).

Pre-norm blocks with RMSNorm, rotary position embeddings (RoPE), grouped-query attention through
PyTorch's scaled_dot_product_attention, SwiGLU feed-forward layers, no biases, and input and
output embeddings tied. Every size comes from ModelConfig, which has no defaults: the YAML config
sets everything.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int  # grouped-query attention: n_heads must be a multiple of it
    ffn_hidden: int  # SwiGLU hidden size
    max_seq_len: int
    rope_theta: float
    norm_eps: float
    init_std: float  # linear layers
    # Tied embeddings make the untrained model repeat its input: the final hidden state is close
    # to the token's own embedding, so that token's logit starts near embed_init_std * d_model
    # (about 10 for 0.02 x 512), and the initial loss would sit well above ln(vocab_size).
    # embed_init_std = 1 / d_model keeps that logit near 1 at any width.
    embed_init_std: float

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("d_model must divide by n_heads, and n_heads by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize in float32 for stability under bf16 autocast
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


def rope_tables(head_dim: int, seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape (seq_len, head_dim // 2)."""
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[2i], x[2i+1]) of x (batch, heads, seq, head_dim) by position angles."""
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    cos, sin = cos[: x.shape[-2]], sin[: x.shape[-2]]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def _qkv(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Queries (batch, heads, seq, head_dim) and keys and values (batch, kv_heads, seq,
        head_dim), with RoPE applied to queries and keys."""
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(b, t, 2, self.n_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Attention output projected back to d_model; causal when mask is None."""
        if self.n_kv_heads != self.n_heads:  # share each key/value head among a group of queries
            group = self.n_heads // self.n_kv_heads
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        b, _, t, _ = q.shape
        return self.out(y.transpose(1, 2).reshape(b, t, -1))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        q, k, v = self._qkv(x, cos, sin)
        return self._attend(q, k, v, None)

    def step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention for new tokens after cached keys and values; returns the output and the
        keys and values including the new tokens. cos and sin are taken at the new positions."""
        q, k, v = self._qkv(x, cos, sin)
        k, v = torch.cat((past_k, k), dim=2), torch.cat((past_v, v), dim=2)
        return self._attend(q, k, v, mask), k, v


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))

    def step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, k, v = self.attn.step(self.attn_norm(x), cos, sin, mask, past_k, past_v)
        x = x + y
        return x + self.ffn(self.ffn_norm(x)), k, v


class DanaLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # scale the layers that write into the residual stream by depth (GPT-2 style)
        for block in self.blocks:
            for layer in (block.attn.out, block.ffn.down):
                nn.init.normal_(layer.weight, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.cfg.init_std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.cfg.embed_init_std)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Logits (batch, seq, vocab) and, with targets, the mean cross-entropy over targets that
        are not -100 (masked positions, e.g. the prompt part of an SFT example)."""
        if idx.shape[1] > self.cfg.max_seq_len:
            raise ValueError(f"sequence of {idx.shape[1]} > max_seq_len {self.cfg.max_seq_len}")
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = F.linear(self.norm(x), self.embed.weight)  # tied output embedding
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),  # targets are often a slice (not contiguous)
                ignore_index=-100,
            )
        return logits, loss

    def step_hidden(
        self, idx: torch.Tensor, positions: torch.Tensor, past: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The KV-cache forward (Phase 7, D-034) up to the final norm. idx (batch, new) are new
        tokens at the absolute positions (new,), shared by the batch; past holds two tensors per
        layer, keys then values, each (batch, kv_heads, past_len, head_dim), and past_len may be
        0. Returns the final hidden states and the keys and values including the new tokens."""
        x = self.embed(idx)
        cos, sin = self.rope_cos[positions], self.rope_sin[positions]
        # a query sees every key up to its own position: explicit, because is_causal would align
        # the mask to the top left and hide the cache from a single new token
        keys = torch.arange(past[0].shape[2] + idx.shape[1], device=idx.device)
        mask = keys[None, :] <= positions[:, None]
        presents = []
        for i, block in enumerate(self.blocks):
            x, k, v = block.step(x, cos, sin, mask, past[2 * i], past[2 * i + 1])
            presents += [k, v]
        return self.norm(x), presents

    def step(
        self, idx: torch.Tensor, positions: torch.Tensor, past: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Logits (batch, new, vocab) of the new tokens, and the updated cache (see step_hidden)."""
        hidden, presents = self.step_hidden(idx, positions, past)
        return F.linear(hidden, self.embed.weight), presents

    def empty_cache(self, batch: int) -> list[torch.Tensor]:
        """A cache with no positions yet, for the first step."""
        shape = (batch, self.cfg.n_kv_heads, 0, self.cfg.head_dim)
        return [self.embed.weight.new_zeros(shape) for _ in range(2 * self.cfg.n_layers)]

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, eos_id: int, temperature: float
    ) -> torch.Tensor:
        """Append up to max_new_tokens tokens to one sequence (batch of 1), stopping at eos_id.
        temperature 0 means greedy decoding. No KV cache: this is for tests and quick checks."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_seq_len :])
            last = logits[:, -1, :].float()
            if temperature == 0:
                nxt = last.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(last / temperature, dim=-1), 1)
            idx = torch.cat((idx, nxt), dim=1)
            if nxt.item() == eos_id:
                break
        return idx


def count_params(model: DanaLM) -> dict[str, int]:
    """Total parameters (the tied embedding counted once) and the non-embedding part."""
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "non_embedding": total - model.embed.weight.numel()}


def flops_per_token(cfg: ModelConfig, seq_len: int) -> int:
    """Training FLOPs per token (forward + backward = 3 x forward, 2 FLOPs per multiply-add):
    all weight matmuls, the tied output projection, and attention scores and values (causal,
    so on average half the sequence is attended)."""
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    per_layer = (
        cfg.d_model * cfg.d_model * 2  # q and out projections
        + cfg.d_model * kv_dim * 2  # k and v
        + cfg.d_model * cfg.ffn_hidden * 3  # SwiGLU gate, up, down
    )
    weights = cfg.n_layers * per_layer + cfg.d_model * cfg.vocab_size
    attention = cfg.n_layers * 2 * cfg.d_model * seq_len // 2  # QK^T and AV, causal half
    return 6 * weights + 6 * attention
