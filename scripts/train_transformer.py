"""Train SmallLM's tokenizer and decoder-only transformer from scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smalllm.backend import TransformerConfig, build_model, save_checkpoint
from smalllm.config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CORPUS_PATH,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_TRAINING_REPORT,
    SPECIAL_TOKENS,
)
from smalllm.retrieval import load_corpus


def train_tokenizer(
    records: list[dict], output: Path, vocab_size: int
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    def text_iterator():
        yield DEFAULT_SYSTEM_PROMPT
        for record in records:
            yield str(record["prompt"])
            yield str(record["response"])

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output), pretty=True)


class InstructionDataset:
    def __init__(self, records, tokenizer, max_seq_len: int) -> None:
        self.examples: list[tuple[list[int], list[int]]] = []
        for record in records:
            prefix = (
                f"<bos><system>\n{DEFAULT_SYSTEM_PROMPT}\n"
                f"<user>\n{record['prompt']}\n<assistant>\n"
            )
            full_text = prefix + str(record["response"]) + "<eos>"
            prefix_ids = tokenizer.encode(prefix).ids
            token_ids = tokenizer.encode(full_text).ids[:max_seq_len]
            if len(token_ids) < 4 or len(prefix_ids) >= len(token_ids):
                continue
            labels = token_ids.copy()
            labels[: min(len(prefix_ids), len(labels))] = [-100] * min(
                len(prefix_ids), len(labels)
            )
            self.examples.append((token_ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def make_collator(pad_id: int):
    import torch

    def collate(batch):
        maximum = max(len(tokens) for tokens, _ in batch)
        input_ids, labels, masks = [], [], []
        for tokens, targets in batch:
            padding = maximum - len(tokens)
            input_ids.append(tokens + [pad_id] * padding)
            labels.append(targets + [-100] * padding)
            masks.append([1] * len(tokens) + [0] * padding)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
        )

    return collate


def loss_for_batch(model, batch, device):
    import torch.nn.functional as functional

    input_ids, labels, attention_mask = (item.to(device) for item in batch)
    logits = model(input_ids, attention_mask)
    return functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def evaluate(model, loader, device, max_batches: int = 50) -> float:
    import torch

    model.eval()
    losses = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            losses.append(float(loss_for_batch(model, batch, device).item()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_TRAINING_REPORT)
    parser.add_argument("--vocab-size", type=int, default=8_000)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1_024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rebuild-tokenizer", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the existing project checkpoint with a fresh optimizer",
    )
    return parser.parse_args()


def main() -> int:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Subset
    from tokenizers import Tokenizer

    args = parse_args()
    if args.resume and args.rebuild_tokenizer:
        raise ValueError("--resume cannot be combined with --rebuild-tokenizer")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    records = load_corpus(args.corpus)
    if args.rebuild_tokenizer or not args.tokenizer.exists():
        print(f"Training {args.vocab_size:,}-token BPE tokenizer from scratch...")
        train_tokenizer(records, args.tokenizer, args.vocab_size)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("trained tokenizer has no padding token")

    dataset = InstructionDataset(records, tokenizer, args.max_seq_len)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    validation_size = max(1, int(len(indices) * 0.05))
    validation = Subset(dataset, indices[:validation_size])
    training = Subset(dataset, indices[validation_size:])
    collator = make_collator(int(pad_id))
    train_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    config = TransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = build_model(config)
    corpus_sha256 = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    initial_step = 0
    previous_history = []
    previous_best_validation = float("inf")
    if args.resume:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if payload.get("config") != config.__dict__:
            raise ValueError("checkpoint architecture does not match training arguments")
        training_state = payload.get("training_state", {})
        recorded_hash = training_state.get("corpus_sha256")
        if recorded_hash and recorded_hash != corpus_sha256:
            raise ValueError("checkpoint was trained on a different corpus")
        model.load_state_dict(payload["model_state"])
        initial_step = int(training_state.get("step", 0))
        if args.report.exists():
            previous_report = json.loads(args.report.read_text(encoding="utf-8"))
            previous_history = list(previous_report.get("history", []))
            previous_best_validation = float(
                previous_report.get("best_validation_loss", float("inf"))
            )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    def learning_rate(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(
            total_steps - args.warmup_steps, 1
        )
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    device = torch.device("cpu")
    model.to(device)
    started = time.perf_counter()
    step = 0
    running_loss = 0.0
    running_count = 0
    best_validation = previous_best_validation
    history = previous_history
    print(
        f"Training {parameter_count:,} parameters on {len(training):,} examples "
        f"for up to {total_steps:,} new steps"
        + (f" (continuing after step {initial_step:,})." if initial_step else ".")
    )
    stop = False
    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            running_loss += float(loss.item())
            running_count += 1
            if step % 10 == 0 or step == 1:
                average = running_loss / running_count
                running_loss = 0.0
                running_count = 0
                elapsed = time.perf_counter() - started
                print(
                    f"step {step:4d}/{total_steps}  loss {average:.4f}  "
                    f"lr {scheduler.get_last_lr()[0]:.2e}  elapsed {elapsed:.1f}s",
                    flush=True,
                )
            if step >= total_steps:
                stop = True
                break
        validation_loss = evaluate(model, validation_loader, device)
        history.append(
            {
                "run_epoch": epoch,
                "step": initial_step + step,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch {epoch} validation loss {validation_loss:.4f} "
            f"(perplexity {math.exp(min(validation_loss, 20)):.2f})"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            save_checkpoint(
                args.checkpoint,
                model,
                config,
                {
                    "step": initial_step + step,
                    "epoch": epoch,
                    "training_examples": len(training),
                    "validation_examples": len(validation),
                    "validation_loss": validation_loss,
                    "parameter_count": parameter_count,
                    "corpus_records": len(records),
                    "seed": args.seed,
                    "corpus_sha256": corpus_sha256,
                },
            )
        if stop:
            break

    report = {
        "trained_from_scratch": True,
        "pretrained_weights_used": False,
        "parameter_count": parameter_count,
        "corpus_records": len(records),
        "training_examples": len(training),
        "validation_examples": len(validation),
        "steps": initial_step + step,
        "run_steps": step,
        "resumed": args.resume,
        "corpus_sha256": corpus_sha256,
        "best_validation_loss": best_validation,
        "best_validation_perplexity": math.exp(min(best_validation, 20)),
        "seconds": time.perf_counter() - started,
        "config": config.__dict__,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint to {args.checkpoint}")
    print(f"Saved training report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
