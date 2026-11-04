"""Train the tiny character-level Nebo language model."""

import argparse
import json
from pathlib import Path

import torch

from model import TinyGPT

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sample_data.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "nebo_model.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")

    text = args.data.read_text(encoding="utf-8")
    if len(text) < 2:
        parser.error(f"Training text must contain at least two characters: {args.data}")
    vocabulary = sorted(set(text))
    stoi = {character: index for index, character in enumerate(vocabulary)}
    encoded = torch.tensor([stoi[character] for character in text], dtype=torch.long)
    args.block_size = min(args.block_size, len(encoded) - 1)

    device = torch.device(args.device)
    encoded = encoded.to(device)
    split = max(args.block_size + 1, int(0.9 * len(encoded)))
    split = min(split, len(encoded))
    train_data = encoded[:split]
    val_data = encoded[split:] if len(encoded) - split > args.block_size else train_data
    config = {"vocab_size": len(vocabulary), "block_size": args.block_size, "n_embd": 96,
              "n_head": 4, "n_layer": 3, "dropout": 0.1}
    model = TinyGPT(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def batch(source):
        max_start = len(source) - args.block_size - 1
        starts = torch.randint(max(1, max_start + 1), (args.batch_size,), device=device)
        x = torch.stack([source[start:start + args.block_size] for start in starts])
        y = torch.stack([source[start + 1:start + args.block_size + 1] for start in starts])
        return x, y

    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch(train_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(val_data)
                _, val_loss = model(vx, vy)
            print(f"step {step:>5}/{args.steps} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")
            model.train()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "model_state": model.cpu().state_dict(), "vocabulary": vocabulary}, args.output)
    print(f"Saved model to {args.output}")
    print("Vocabulary size:", len(vocabulary))
    print("Model config:", json.dumps(config, ensure_ascii=False))


if __name__ == "__main__":
    main()
