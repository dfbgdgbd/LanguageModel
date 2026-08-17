"""Run a broad smoke evaluation over retrieval, tools, and optional generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smalllm import AdvancedAssistant, AssistantSettings


SCENARIOS = [
    "Explain how photosynthesis stores energy in simple terms.",
    "Write a Python function that removes duplicate strings while preserving order.",
    "Help me structure a professional project update email.",
    "Compare REST and GraphQL APIs.",
    "Give me a three-day study plan for a biology exam.",
    "Explain why database indexes speed up some queries.",
    "Brainstorm five names for an environmentally friendly delivery service.",
    "What should I check first when debugging an HTTP 500 error?",
    "Summarize the difference between weather and climate.",
    "Explain compound interest to a beginner.",
    "What is 18.5 * (7 - 2)?",
    "Convert 32 fahrenheit to celsius.",
    "Count words in: A small model can still solve useful tasks.",
    "Validate JSON: {\"name\": \"SmallLM\", \"ready\": true}",
    "What is today's date?",
    "Create a checklist for launching a small website.",
    "Explain recursion with a tiny example.",
    "How can I recognize a phishing message?",
    "Suggest a clear outline for a short science report.",
    "What are the tradeoffs between a list and a set in Python?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["retrieval", "hybrid", "transformer"], default="retrieval"
    )
    parser.add_argument("--limit", type=int, default=len(SCENARIOS))
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "evaluation_report.json"
    )
    args = parser.parse_args()
    assistant = AdvancedAssistant(AssistantSettings(backend=args.backend))
    report = []
    for query in SCENARIOS[: args.limit]:
        started = time.perf_counter()
        response = assistant.respond(query)
        elapsed = time.perf_counter() - started
        report.append(
            {
                "query": query,
                "response": response.text,
                "backend": response.backend,
                "seconds": round(elapsed, 3),
                "top_retrieval_score": round(response.retrieved[0].score, 4)
                if response.retrieved
                else None,
            }
        )
        print(f"[{response.backend:>18}] {elapsed:6.2f}s  {query}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(report)} results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
