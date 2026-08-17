"""A small, dependency-free character n-gram language model.

The model learns how often each character follows contexts found in a text
corpus. During generation it backs off to shorter contexts when it encounters
something it did not see during training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any


MODEL_VERSION = 1


class CharacterNGramLanguageModel:
    """A character-level n-gram model with suffix backoff."""

    def __init__(self, order: int = 5) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.transitions: dict[str, Counter[str]] = {}
        self.character_count = 0

    @property
    def vocabulary(self) -> list[str]:
        """Return every character learned by the model in stable order."""
        root_counts = self.transitions.get("", Counter())
        return sorted(root_counts)

    def train(self, text: str) -> None:
        """Learn character transitions from *text*, replacing prior training."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text:
            raise ValueError("training text cannot be empty")

        learned: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for index, character in enumerate(text):
            max_width = min(self.order, index)
            for width in range(max_width + 1):
                context = text[index - width : index] if width else ""
                learned[context][character] += 1


        self.transitions = dict(learned)
        self.character_count = len(text)

    def _counts_for(self, context: str) -> Counter[str]:
        """Find the longest known suffix of *context*."""
        for width in range(min(self.order, len(context)), -1, -1):
            suffix = context[-width:] if width else ""
            counts = self.transitions.get(suffix)
            if counts:
                return counts
        raise ValueError("model has not been trained")

    @staticmethod
    def _sample(
        counts: Counter[str], temperature: float, generator: random.Random
    ) -> str:
        if temperature < 0:
            raise ValueError("temperature cannot be negative")

        choices = sorted(counts)
        if temperature == 0:
            return min(choices, key=lambda item: (-counts[item], item))

        weights = [counts[item] ** (1.0 / temperature) for item in choices]
        return generator.choices(choices, weights=weights, k=1)[0]

    def generate(
        self,
        prompt: str = "",
        length: int = 300,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> str:
        """Continue *prompt* with exactly *length* sampled characters."""
        if not self.transitions:
            raise ValueError("model has not been trained")
        if length < 0:
            raise ValueError("length cannot be negative")

        generator = random.Random(seed)
        output = prompt
        for _ in range(length):
            counts = self._counts_for(output)
            output += self._sample(counts, temperature, generator)
        return output

    def to_dict(self) -> dict[str, Any]:
        """Convert the trained model to a JSON-compatible dictionary."""
        if not self.transitions:
            raise ValueError("model has not been trained")
        return {
            "version": MODEL_VERSION,
            "type": "character_ngram",
            "order": self.order,
            "character_count": self.character_count,
            "vocabulary": self.vocabulary,
            "transitions": {
                context: dict(sorted(counts.items()))
                for context, counts in sorted(self.transitions.items())
            },
        }

    def save(self, path: str | Path) -> None:
        """Save a trained model as readable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CharacterNGramLanguageModel:
        """Load and validate a model saved by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != MODEL_VERSION:
            raise ValueError("unsupported model version")
        if payload.get("type") != "character_ngram":
            raise ValueError("file is not a character n-gram model")

        model = cls(order=int(payload["order"]))
        model.character_count = int(payload["character_count"])
        raw_transitions = payload.get("transitions")
        if not isinstance(raw_transitions, dict) or "" not in raw_transitions:
            raise ValueError("model contains no transitions")

        model.transitions = {
            str(context): Counter(
                {str(character): int(count) for character, count in counts.items()}
            )
            for context, counts in raw_transitions.items()
        }
        return model


def train_command(args: argparse.Namespace) -> None:
    corpus_path = Path(args.data)
    text = corpus_path.read_text(encoding="utf-8")
    model = CharacterNGramLanguageModel(order=args.order)
    model.train(text)
    model.save(args.model)
    print(
        f"Trained order-{model.order} model on {model.character_count:,} "
        f"characters ({len(model.vocabulary)} unique)."
    )
    print(f"Saved model to {Path(args.model).resolve()}")


def generate_command(args: argparse.Namespace) -> None:
    model = CharacterNGramLanguageModel.load(args.model)
    print(
        model.generate(
            prompt=args.prompt,
            length=args.length,
            temperature=args.temperature,
            seed=args.seed,
        )
    )


def info_command(args: argparse.Namespace) -> None:
    model = CharacterNGramLanguageModel.load(args.model)
    print(f"Type: character n-gram")
    print(f"Order: {model.order}")
    print(f"Training characters: {model.character_count:,}")
    print(f"Vocabulary size: {len(model.vocabulary)}")
    print(f"Learned contexts: {len(model.transitions):,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a small character n-gram language model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a model from a text file")
    train_parser.add_argument("--data", default="data/training.txt", help="UTF-8 corpus")
    train_parser.add_argument("--model", default="model.json", help="output model")
    train_parser.add_argument(
        "--order", type=int, default=5, help="maximum context length (default: 5)"
    )
    train_parser.set_defaults(handler=train_command)

    generate_parser = subparsers.add_parser(
        "generate", help="generate text from a trained model"
    )
    generate_parser.add_argument("--model", default="model.json", help="saved model")
    generate_parser.add_argument("--prompt", default="", help="text to continue")
    generate_parser.add_argument(
        "--length", type=int, default=300, help="characters to generate"
    )
    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="0 is deterministic; larger values add variety",
    )
    generate_parser.add_argument("--seed", type=int, default=None, help="random seed")
    generate_parser.set_defaults(handler=generate_command)

    info_parser = subparsers.add_parser("info", help="show saved model statistics")
    info_parser.add_argument("--model", default="model.json", help="saved model")
    info_parser.set_defaults(handler=info_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
