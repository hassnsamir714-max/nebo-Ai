
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE,
    MAX_SEQ_LEN,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    D_FF
)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            D_MODEL,
            N_HEADS,
            batch_first=True
        )

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN),
                diagonal=1
            ).bool()
        )

    def forward(self, x):
        seq_len = x.size(1)

        mask = self.mask[:seq_len, :seq_len]

        output, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            need_weights=False
        )

        return output


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)

        self.attention = CausalSelfAttention()

        self.feed_forward = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class NeboAI(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE, D_MODEL
        )

        self.position_embedding = nn.Embedding(
            MAX_SEQ_LEN, D_MODEL
        )

        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(N_LAYERS)]
        )

        self.norm = nn.LayerNorm(D_MODEL)

        self.output = nn.Linear(
            D_MODEL, VOCAB_SIZE, bias=False
        )

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        x = self.blocks(x)
        x = self.norm(x)

        logits = self.output(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss


if __name__ == "__main__":
    model = NeboAI()

    sample = torch.randint(
        0, VOCAB_SIZE, (2, 32)
    )

    logits, loss = model(sample, sample)

    print("Nebo AI Model initialized")
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))
    print("Output shape:", logits.shape)
      
