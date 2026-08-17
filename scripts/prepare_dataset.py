"""Build a filtered instruction corpus from curated data and OpenAssistant.

The resulting JSONL is deterministic for a given upstream dataset revision and
contains only English, top-ranked, reviewed prompt-response pairs that pass
quality, toxicity, PII, and length filters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENSITIVE_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\b(?:api[_ -]?key|password)\s*(?:is|=|:)\s*\S+", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
]


def label_values(row: dict[str, Any]) -> dict[str, float]:
    labels = row.get("labels") or {}
    return {
        str(name): float(value or 0)
        for name, value in zip(labels.get("name", []), labels.get("value", []))
    }


def contains_sensitive_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def safe_pair(prompt: dict[str, Any], answer: dict[str, Any]) -> bool:
    prompt_text = str(prompt.get("text", "")).strip()
    answer_text = str(answer.get("text", "")).strip()
    labels = label_values(answer)
    detoxify = answer.get("detoxify") or {}
    return bool(
        answer.get("role") == "assistant"
        and answer.get("lang") == "en"
        and answer.get("rank") == 0
        and not answer.get("deleted")
        and answer.get("tree_state") == "ready_for_export"
        and prompt.get("role") == "prompter"
        and prompt.get("lang") == "en"
        and not prompt.get("deleted")
        and 4 <= len(prompt_text) <= 1_200
        and 20 <= len(answer_text) <= 3_000
        and float(detoxify.get("toxicity", 0) or 0) < 0.15
        and float(detoxify.get("sexual_explicit", 0) or 0) < 0.08
        and labels.get("not_appropriate", 0) <= 0.25
        and labels.get("pii", 0) <= 0.25
        and labels.get("hate_speech", 0) <= 0.25
        and labels.get("sexual_content", 0) <= 0.25
        and labels.get("violence", 0) <= 0.5
        and labels.get("quality", 1) >= 0.5
        and not contains_sensitive_text(prompt_text)
        and not contains_sensitive_text(answer_text)
    )


def curated_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for topic_index, topic in enumerate(payload):
        responses = topic["responses"]
        for prompt_index, prompt in enumerate(topic["prompts"]):
            records.append(
                {
                    "id": f"curated-{topic_index:03d}-{prompt_index:03d}",
                    "prompt": prompt.strip(),
                    "response": responses[prompt_index % len(responses)].strip(),
                    "source": "smalllm_curated",
                    "topic": topic["topic"],
                    "license": "Apache-2.0",
                }
            )
    return records


def openassistant_records(
    dataset_name: str, source_prefix: str, limit: int
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train")
    messages = {row["message_id"]: row for row in dataset}
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for answer in dataset:
        parent = messages.get(answer.get("parent_id"))
        if parent is None or not safe_pair(parent, answer):
            continue
        prompt_text = str(parent["text"]).strip()
        answer_text = str(answer["text"]).strip()
        fingerprint = (prompt_text.casefold(), answer_text.casefold())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        records.append(
            {
                "id": f"{source_prefix}-{answer['message_id']}",
                "prompt": prompt_text,
                "response": answer_text,
                "source": dataset_name,
                "topic": "open_domain",
                "license": "Apache-2.0",
            }
        )
    records.sort(key=lambda item: item["id"])
    return records[:limit] if limit else records


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first occurrence of an identical prompt/response pair."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        fingerprint = (
            str(record["prompt"]).strip().casefold(),
            str(record["response"]).strip().casefold(),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(record)
    return unique


def dolly_records(limit: int) -> list[dict[str, Any]]:
    """Load human-written Databricks Dolly instructions for retrieval coverage."""
    from datasets import load_dataset

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        prompt = str(row.get("instruction", "")).strip()
        response = str(row.get("response", "")).strip()
        if not (
            4 <= len(prompt) <= 1_200
            and 20 <= len(response) <= 3_000
            and not contains_sensitive_text(prompt)
            and not contains_sensitive_text(response)
        ):
            continue
        records.append(
            {
                "id": f"dolly-{index:05d}",
                "prompt": prompt,
                "response": response,
                "source": "databricks/databricks-dolly-15k",
                "topic": str(row.get("category") or "open_domain"),
                "license": "CC-BY-SA-3.0",
            }
        )
        if limit and len(records) >= limit:
            break
    return records


def write_jsonl(records: list[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return output.stat().st_size


def build_dataset(
    curated: Path,
    output: Path,
    retrieval_output: Path,
    oasst1_limit: int,
    oasst2_limit: int,
    dolly_limit: int,
) -> tuple[int, int, int, int, int, int]:
    local = curated_records(curated)
    oasst1 = openassistant_records(
        "OpenAssistant/oasst1", "oasst1", oasst1_limit
    )
    oasst2 = openassistant_records(
        "OpenAssistant/oasst2", "oasst2", oasst2_limit
    )
    combined = deduplicate([*local, *oasst1, *oasst2])
    training_size = write_jsonl(combined, output)
    knowledge = deduplicate([*combined, *dolly_records(dolly_limit)])
    retrieval_size = write_jsonl(knowledge, retrieval_output)
    source_counts = {
        source: sum(record["source"] == source for record in combined)
        for source in {
            "smalllm_curated",
            "OpenAssistant/oasst1",
            "OpenAssistant/oasst2",
        }
    }
    return (
        source_counts["smalllm_curated"],
        source_counts["OpenAssistant/oasst1"],
        source_counts["OpenAssistant/oasst2"],
        len(knowledge) - len(combined),
        training_size,
        retrieval_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curated", type=Path, default=PROJECT_ROOT / "data" / "instructions.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "instructions_large.jsonl",
    )
    parser.add_argument(
        "--retrieval-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "knowledge_large.jsonl",
    )
    parser.add_argument(
        "--oasst1-limit",
        type=int,
        default=0,
        help="maximum filtered OASST1 pairs; 0 keeps every filtered pair",
    )
    parser.add_argument(
        "--oasst2-limit",
        type=int,
        default=0,
        help="maximum filtered OASST2 pairs; 0 keeps every filtered pair",
    )
    parser.add_argument(
        "--dolly-limit",
        type=int,
        default=0,
        help="maximum filtered Dolly retrieval pairs; 0 keeps every filtered pair",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.oasst1_limit < 0 or args.oasst2_limit < 0 or args.dolly_limit < 0:
        print("dataset limits cannot be negative", file=sys.stderr)
        return 2
    local, oasst1, oasst2, dolly, training_size, retrieval_size = build_dataset(
        args.curated,
        args.output,
        args.retrieval_output,
        args.oasst1_limit,
        args.oasst2_limit,
        args.dolly_limit,
    )
    print(f"Wrote {local + oasst1 + oasst2:,} instruction pairs to {args.output}")
    print(f"  curated: {local:,}")
    print(f"  OASST1: {oasst1:,}")
    print(f"  OASST2: {oasst2:,}")
    print(f"  training size: {training_size / 1_048_576:.2f} MiB")
    print(
        f"Wrote {local + oasst1 + oasst2 + dolly:,} retrieval pairs "
        f"to {args.retrieval_output}"
    )
    print(f"  Dolly: {dolly:,}")
    print(f"  retrieval size: {retrieval_size / 1_048_576:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
