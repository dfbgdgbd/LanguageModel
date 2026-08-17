"""A dependency-free, trainable query-response language model.

The project combines two small models:

* a TF-IDF instruction retriever for useful, grounded query responses; and
* a character n-gram model for creative text continuation.

It is intentionally educational and runs with only the Python standard library.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any


CHARACTER_MODEL_VERSION = 1
ASSISTANT_MODEL_VERSION = 2
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "please",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
}


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
        root_counts = self.transitions.get("", Counter())
        return sorted(root_counts)

    def train(self, text: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text:
            raise ValueError("training text cannot be empty")

        learned: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for index, character in enumerate(text):
            for width in range(min(self.order, index) + 1):
                context = text[index - width : index] if width else ""
                learned[context][character] += 1
        self.transitions = dict(learned)
        self.character_count = len(text)

    def _counts_for(self, context: str) -> Counter[str]:
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
        if not self.transitions:
            raise ValueError("model has not been trained")
        if length < 0:
            raise ValueError("length cannot be negative")

        generator = random.Random(seed)
        output = prompt
        for _ in range(length):
            output += self._sample(
                self._counts_for(output), temperature, generator
            )
        return output

    def to_dict(self) -> dict[str, Any]:
        if not self.transitions:
            raise ValueError("model has not been trained")
        return {
            "version": CHARACTER_MODEL_VERSION,
            "type": "character_ngram",
            "order": self.order,
            "character_count": self.character_count,
            "vocabulary": self.vocabulary,
            "transitions": {
                context: dict(sorted(counts.items()))
                for context, counts in sorted(self.transitions.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CharacterNGramLanguageModel:
        if payload.get("version") != CHARACTER_MODEL_VERSION:
            raise ValueError("unsupported character model version")
        if payload.get("type") != "character_ngram":
            raise ValueError("file is not a character n-gram model")

        model = cls(order=int(payload["order"]))
        model.character_count = int(payload["character_count"])
        raw_transitions = payload.get("transitions")
        if not isinstance(raw_transitions, dict) or "" not in raw_transitions:
            raise ValueError("model contains no character transitions")
        model.transitions = {
            str(context): Counter(
                {str(character): int(count) for character, count in counts.items()}
            )
            for context, counts in raw_transitions.items()
        }
        return model

    def save(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> CharacterNGramLanguageModel:
        return cls.from_dict(_read_json(path))


def _stem(word: str) -> str:
    """Apply a deliberately small stemmer to improve prompt matching."""
    for suffix in ("ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            if suffix == "ies":
                return word[:-3] + "y"
            return word[: -len(suffix)]
    return word


def _features(text: str) -> Counter[str]:
    words = [_stem(word) for word in WORD_RE.findall(text.lower())]
    content = [word for word in words if word not in STOP_WORDS]
    if not content:
        content = words

    features: Counter[str] = Counter(f"word:{word}" for word in content)
    # Phrase features retain useful structure such as "can_you" even when one
    # of the words is too common to be a useful unigram.
    for first, second in zip(words, words[1:]):
        features[f"pair:{first}_{second}"] += 1
    return features


def _make_vector(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    weighted = {
        feature: (1.0 + math.log(count)) * idf[feature]
        for feature, count in counts.items()
        if feature in idf
    }
    magnitude = math.sqrt(sum(value * value for value in weighted.values()))
    if not magnitude:
        return {}
    return {feature: value / magnitude for feature, value in weighted.items()}


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(feature, 0.0) for feature, value in left.items())


class QueryAssistantModel:
    """A trained instruction retriever with a creative n-gram fallback model."""

    def __init__(self, order: int = 5) -> None:
        self.generator = CharacterNGramLanguageModel(order=order)
        self.intents: list[dict[str, Any]] = []
        self.examples: list[dict[str, Any]] = []
        self.idf: dict[str, float] = {}

    def train(self, corpus: str, instructions: list[dict[str, Any]]) -> None:
        if not instructions:
            raise ValueError("instruction data cannot be empty")

        prompts: list[tuple[int, str]] = []
        validated: list[dict[str, Any]] = []
        conversation_text: list[str] = [corpus]
        for intent_index, record in enumerate(instructions):
            topic = str(record.get("topic", "")).strip()
            raw_prompts = record.get("prompts")
            raw_responses = record.get("responses")
            if not topic or not isinstance(raw_prompts, list) or not raw_prompts:
                raise ValueError(f"instruction {intent_index} needs a topic and prompts")
            if not isinstance(raw_responses, list) or not raw_responses:
                raise ValueError(f"instruction {topic!r} needs responses")

            intent_prompts = [str(item).strip() for item in raw_prompts if str(item).strip()]
            responses = [str(item).strip() for item in raw_responses if str(item).strip()]
            if not intent_prompts or not responses:
                raise ValueError(f"instruction {topic!r} contains empty training data")

            validated.append({"topic": topic, "responses": responses})
            for prompt_index, prompt in enumerate(intent_prompts):
                prompts.append((intent_index, prompt))
                response = responses[prompt_index % len(responses)]
                conversation_text.append(f"User: {prompt}\nAssistant: {response}")

        document_features = [_features(prompt) for _, prompt in prompts]
        document_frequency: Counter[str] = Counter()
        for counts in document_features:
            document_frequency.update(counts.keys())
        document_count = len(document_features)
        self.idf = {
            feature: math.log((document_count + 1) / (frequency + 1)) + 1.0
            for feature, frequency in sorted(document_frequency.items())
        }

        self.intents = validated
        self.examples = [
            {
                "intent": intent_index,
                "prompt": prompt,
                "vector": _make_vector(counts, self.idf),
            }
            for (intent_index, prompt), counts in zip(prompts, document_features)
        ]
        self.generator.train("\n\n".join(conversation_text))

    def match(self, query: str) -> tuple[int | None, float]:
        if not self.examples:
            raise ValueError("assistant model has not been trained")
        query_features = _features(query)
        # Phrase-only overlap such as "how_do" is not enough evidence. At
        # least one meaningful query word must exist in the trained vocabulary.
        has_known_word = any(
            feature.startswith("word:") and feature in self.idf
            for feature in query_features
        )
        if not has_known_word:
            return None, 0.0
        query_vector = _make_vector(query_features, self.idf)
        if not query_vector:
            return None, 0.0

        normalized_query = " ".join(WORD_RE.findall(query.lower()))
        query_word_count = sum(
            feature.startswith("word:") for feature in query_vector
        )
        best_intent: int | None = None
        best_score = 0.0
        for example in self.examples:
            normalized_example = " ".join(
                WORD_RE.findall(str(example["prompt"]).lower())
            )
            if normalized_query == normalized_example:
                return int(example["intent"]), 1.0
            score = _dot(query_vector, example["vector"])
            example_word_count = sum(
                feature.startswith("word:") for feature in example["vector"]
            )
            if query_word_count > 1 and example_word_count == 1:
                score *= 0.6
            if score > best_score:
                best_intent = int(example["intent"])
                best_score = score
        return best_intent, best_score

    def answer(
        self,
        query: str,
        threshold: float = 0.24,
        seed: int | None = None,
    ) -> tuple[str, str | None, float]:
        query = query.strip()
        if not query:
            return "Please enter a question or request.", None, 0.0
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be between 0 and 1")

        calculation = _try_calculation(query)
        if calculation is not None:
            return f"The result is {calculation}.", "calculator", 1.0

        intent_index, score = self.match(query)
        if intent_index is None or score < threshold:
            return (
                "I do not have enough matching training data to answer that "
                "reliably. Try rephrasing it, or add examples to "
                "data/instructions.json and retrain the model.",
                None,
                score,
            )

        intent = self.intents[intent_index]
        responses = intent["responses"]
        if seed is None:
            digest = hashlib.sha256(query.lower().encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
        response = random.Random(seed).choice(responses)
        return response, str(intent["topic"]), score

    def to_dict(self) -> dict[str, Any]:
        if not self.examples:
            raise ValueError("assistant model has not been trained")
        return {
            "version": ASSISTANT_MODEL_VERSION,
            "type": "hybrid_query_assistant",
            "instruction_examples": len(self.examples),
            "feature_count": len(self.idf),
            "intents": self.intents,
            "idf": self.idf,
            "examples": self.examples,
            "generator": self.generator.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QueryAssistantModel:
        if payload.get("version") != ASSISTANT_MODEL_VERSION:
            raise ValueError("unsupported assistant model version")
        if payload.get("type") != "hybrid_query_assistant":
            raise ValueError("file is not a hybrid query assistant model")

        generator = CharacterNGramLanguageModel.from_dict(payload["generator"])
        model = cls(order=generator.order)
        model.generator = generator
        model.intents = list(payload["intents"])
        model.idf = {str(key): float(value) for key, value in payload["idf"].items()}
        model.examples = [
            {
                "intent": int(example["intent"]),
                "prompt": str(example["prompt"]),
                "vector": {
                    str(key): float(value)
                    for key, value in example["vector"].items()
                },
            }
            for example in payload["examples"]
        ]
        if not model.intents or not model.examples:
            raise ValueError("assistant model contains no instruction examples")
        return model

    def save(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> QueryAssistantModel:
        return cls.from_dict(_read_json(path))


def _safe_number(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_number(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_number(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _safe_number(node.left)
        right = _safe_number(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > 10 or abs(left) > 1_000_000:
                raise ValueError("exponent is too large")
            result = left**right
        else:
            raise ValueError("unsupported arithmetic operator")
        if type(result) not in (int, float):
            raise ValueError("calculation must produce a real number")
        if not math.isfinite(float(result)) or abs(result) > 1e15:
            raise ValueError("calculation result is too large")
        return result
    raise ValueError("unsupported arithmetic expression")


def _try_calculation(query: str) -> str | None:
    expression = query.lower().strip().rstrip("?")
    expression = re.sub(
        r"^(?:what is|calculate|compute|evaluate|solve)\s+", "", expression
    )
    replacements = {
        "divided by": "/",
        "multiplied by": "*",
        "times": "*",
        "plus": "+",
        "minus": "-",
        "×": "*",
        "÷": "/",
    }
    for phrase, operator in replacements.items():
        expression = expression.replace(phrase, operator)
    if not re.fullmatch(r"[0-9eE.\s+\-*/%()]+", expression):
        return None
    if not re.search(r"[+\-*/%]", expression):
        return None

    try:
        value = _safe_number(ast.parse(expression, mode="eval"))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model JSON must contain an object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_instructions(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("instruction JSON must contain a list")
    return payload


def _load_generator(path: str | Path) -> CharacterNGramLanguageModel:
    payload = _read_json(path)
    if payload.get("type") == "hybrid_query_assistant":
        return QueryAssistantModel.from_dict(payload).generator
    return CharacterNGramLanguageModel.from_dict(payload)


def train_command(args: argparse.Namespace) -> None:
    corpus = Path(args.data).read_text(encoding="utf-8")
    instructions = _load_instructions(args.instructions)
    model = QueryAssistantModel(order=args.order)
    model.train(corpus, instructions)
    model.save(args.model)
    print(
        f"Trained on {len(model.examples):,} instruction examples and "
        f"{model.generator.character_count:,} total characters."
    )
    print(
        f"Learned {len(model.intents):,} response topics, "
        f"{len(model.idf):,} retrieval features, and "
        f"{len(model.generator.transitions):,} character contexts."
    )
    print(f"Saved model to {Path(args.model).resolve()}")


def ask_command(args: argparse.Namespace) -> None:
    model = QueryAssistantModel.load(args.model)
    answer, topic, score = model.answer(
        args.prompt, threshold=args.threshold, seed=args.seed
    )
    print(answer)
    if args.show_score:
        print(f"[topic={topic or 'unknown'}, confidence={score:.3f}]")


def chat_command(args: argparse.Namespace) -> None:
    model = QueryAssistantModel.load(args.model)
    print("SmallLM chat is ready. Type 'quit' or 'exit' to stop.")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"quit", "exit"}:
            break
        answer, _, _ = model.answer(query, threshold=args.threshold)
        print(f"SmallLM: {answer}")


def generate_command(args: argparse.Namespace) -> None:
    model = _load_generator(args.model)
    print(
        model.generate(
            prompt=args.prompt,
            length=args.length,
            temperature=args.temperature,
            seed=args.seed,
        )
    )


def info_command(args: argparse.Namespace) -> None:
    payload = _read_json(args.model)
    if payload.get("type") == "hybrid_query_assistant":
        model = QueryAssistantModel.from_dict(payload)
        print("Type: hybrid query assistant")
        print(f"Response topics: {len(model.intents):,}")
        print(f"Instruction examples: {len(model.examples):,}")
        print(f"Retrieval features: {len(model.idf):,}")
        generator = model.generator
    else:
        generator = CharacterNGramLanguageModel.from_dict(payload)
        print("Type: character n-gram")
    print(f"Generator order: {generator.order}")
    print(f"Generator training characters: {generator.character_count:,}")
    print(f"Generator vocabulary size: {len(generator.vocabulary):,}")
    print(f"Generator contexts: {len(generator.transitions):,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a small query-response language model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train all model components")
    train_parser.add_argument("--data", default="data/training.txt", help="text corpus")
    train_parser.add_argument(
        "--instructions",
        default="data/instructions.json",
        help="instruction-response dataset",
    )
    train_parser.add_argument("--model", default="model.json", help="output model")
    train_parser.add_argument("--order", type=int, default=5, help="n-gram order")
    train_parser.set_defaults(handler=train_command)

    ask_parser = subparsers.add_parser("ask", help="answer one query")
    ask_parser.add_argument("--model", default="model.json", help="saved model")
    ask_parser.add_argument("--prompt", required=True, help="question or request")
    ask_parser.add_argument(
        "--threshold", type=float, default=0.24, help="minimum match confidence"
    )
    ask_parser.add_argument("--seed", type=int, default=None, help="response seed")
    ask_parser.add_argument(
        "--show-score", action="store_true", help="show matched topic and confidence"
    )
    ask_parser.set_defaults(handler=ask_command)

    chat_parser = subparsers.add_parser("chat", help="start an interactive chat")
    chat_parser.add_argument("--model", default="model.json", help="saved model")
    chat_parser.add_argument(
        "--threshold", type=float, default=0.24, help="minimum match confidence"
    )
    chat_parser.set_defaults(handler=chat_command)

    generate_parser = subparsers.add_parser(
        "generate", help="use the creative character generator"
    )
    generate_parser.add_argument("--model", default="model.json", help="saved model")
    generate_parser.add_argument("--prompt", default="", help="text to continue")
    generate_parser.add_argument("--length", type=int, default=300)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--seed", type=int, default=None)
    generate_parser.set_defaults(handler=generate_command)

    info_parser = subparsers.add_parser("info", help="show model statistics")
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
