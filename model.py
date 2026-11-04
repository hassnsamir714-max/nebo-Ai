"""A small character-level causal Transformer implemented with PyTorch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, steps, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, steps, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, steps, self.n_head, self.head_size).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.causal_mask[:steps, :steps] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.feed_forward(self.ln2(x))


class TinyGPT(nn.Module):
    """A teaching model that predicts the next character in a text sequence."""

    def __init__(self, vocab_size: int, block_size: int = 128, n_embd: int = 96,
                 n_head: int = 4, n_layer: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*(TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)))
        self.final_norm = nn.LayerNorm(n_embd)
        self.language_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, indices: torch.Tensor, targets: torch.Tensor | None = None):
        _, steps = indices.shape
        if steps > self.block_size:
            indices = indices[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]
            steps = self.block_size
        positions = torch.arange(steps, device=indices.device)
        x = self.token_embedding(indices) + self.position_embedding(positions)
        logits = self.language_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, indices: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        temperature = max(float(temperature), 0.05)
        for _ in range(max_new_tokens):
            logits, _ = self(indices[:, -self.block_size:])
            probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            indices = torch.cat((indices, torch.multinomial(probabilities, num_samples=1)), dim=1)
        return indices
