"""Command-line interface for SmallLM's from-scratch language model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def prepare_data_command(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "prepare_dataset.py"),
            "--oasst1-limit",
            str(args.oasst1_limit),
            "--oasst2-limit",
            str(args.oasst2_limit),
            "--dolly-limit",
            str(args.dolly_limit),
        ],
        check=True,
    )


def train_command(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent
    subprocess.run(
        [sys.executable, str(root / "scripts" / "build_index.py")], check=True
    )
    command = [
        sys.executable,
        str(root / "scripts" / "train_transformer.py"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]
    if args.max_steps:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.rebuild_tokenizer:
        command.append("--rebuild-tokenizer")
    if args.resume:
        command.append("--resume")
    subprocess.run(command, check=True)


def assistant_from_args(args: argparse.Namespace):
    from smalllm import AdvancedAssistant, AssistantSettings
    from smalllm.backend import GenerationSettings

    return AdvancedAssistant(
        AssistantSettings(
            backend=args.backend,
            direct_retrieval_threshold=args.threshold,
            seed=args.seed,
            generation=GenerationSettings(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            ),
        )
    )


def ask_command(args: argparse.Namespace) -> None:
    response = assistant_from_args(args).respond(args.prompt)
    print(response.text)
    if args.show_score:
        score = response.retrieved[0].score if response.retrieved else 0.0
        print(f"[backend={response.backend}, retrieval_score={score:.3f}]")


def chat_command(args: argparse.Namespace) -> None:
    assistant = assistant_from_args(args)
    print("SmallLM from-scratch chat is ready. Type 'quit' or 'exit' to stop.")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"quit", "exit"}:
            break
        response = assistant.respond(query)
        print(f"SmallLM: {response.text}")
        if args.show_backend:
            print(f"  [{response.backend}]")


def generate_command(args: argparse.Namespace) -> None:
    from smalllm.backend import GenerationSettings, ScratchTransformerBackend

    model = ScratchTransformerBackend(
        settings=GenerationSettings(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ),
        seed=args.seed,
    )
    print(model.generate(args.prompt))


def info_command(_: argparse.Namespace) -> None:
    from smalllm.backend import ScratchTransformerBackend
    from smalllm.config import (
        DEFAULT_CORPUS_PATH,
        DEFAULT_RETRIEVAL_CORPUS_PATH,
        DEFAULT_TRAINING_REPORT,
    )
    from smalllm.retrieval import load_corpus

    model = ScratchTransformerBackend()
    report = json.loads(DEFAULT_TRAINING_REPORT.read_text(encoding="utf-8"))
    print("Type: decoder-only transformer trained from scratch")
    print("Pretrained weights used: no")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Tokenizer vocabulary: {model.tokenizer.vocab_size:,}")
    print(f"Context length: {model.config.max_seq_len:,}")
    print(f"Instruction pairs: {len(load_corpus(DEFAULT_CORPUS_PATH)):,}")
    print(f"Retrieval scenarios: {len(load_corpus(DEFAULT_RETRIEVAL_CORPUS_PATH)):,}")
    print(f"Training steps: {report['steps']:,}")
    print(f"Validation loss: {report['best_validation_loss']:.4f}")
    print(f"Validation perplexity: {report['best_validation_perplexity']:.2f}")


def add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)


def add_assistant_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["hybrid", "transformer", "retrieval"],
        default="hybrid",
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    add_generation_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run SmallLM, a decoder-only transformer built from scratch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-data", help="download, filter, and assemble training data"
    )
    prepare_parser.add_argument("--oasst1-limit", type=int, default=0)
    prepare_parser.add_argument("--oasst2-limit", type=int, default=0)
    prepare_parser.add_argument("--dolly-limit", type=int, default=0)
    prepare_parser.set_defaults(handler=prepare_data_command)

    train_parser = subparsers.add_parser(
        "train", help="train retrieval and transformer components from scratch"
    )
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=12)
    train_parser.add_argument("--max-steps", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--rebuild-tokenizer", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.set_defaults(handler=train_command)

    ask_parser = subparsers.add_parser("ask", help="answer one query")
    ask_parser.add_argument("--prompt", required=True, help="question or request")
    add_assistant_options(ask_parser)
    ask_parser.add_argument("--show-score", action="store_true")
    ask_parser.set_defaults(handler=ask_command)

    chat_parser = subparsers.add_parser("chat", help="start an interactive chat")
    add_assistant_options(chat_parser)
    chat_parser.add_argument("--show-backend", action="store_true")
    chat_parser.set_defaults(handler=chat_command)

    generate_parser = subparsers.add_parser(
        "generate", help="generate directly with the trained transformer"
    )
    generate_parser.add_argument("--prompt", default="", help="text to continue")
    add_generation_options(generate_parser)
    generate_parser.set_defaults(handler=generate_command)

    info_parser = subparsers.add_parser("info", help="show model statistics")
    info_parser.set_defaults(handler=info_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
