"""Hybrid orchestration for the new from-scratch SmallLM model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal

from .backend import GenerationSettings, ScratchTransformerBackend
from .config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MAX_HISTORY_MESSAGES,
    DEFAULT_RETRIEVAL_INDEX,
    DEFAULT_RETRIEVAL_CORPUS_PATH,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOKENIZER_PATH,
)
from .retrieval import RetrievedDocument, TrainedRetriever
from .tools import run_tool


BackendName = Literal["hybrid", "transformer", "retrieval"]

GENERATION_STOPWORDS = {
    "about",
    "answer",
    "could",
    "explain",
    "give",
    "help",
    "make",
    "please",
    "short",
    "tell",
    "that",
    "this",
    "what",
    "with",
    "write",
    "would",
    "your",
}


def generation_is_grounded(query: str, text: str) -> bool:
    """Reject empty, repetitive, or topically disconnected generations."""
    words = re.findall(r"[a-z][a-z0-9'-]+", text.casefold())
    if len(words) < 6 or len(set(words)) < max(4, len(words) // 5):
        return False
    query_terms = {
        word
        for word in re.findall(r"[a-z][a-z0-9'-]+", query.casefold())
        if len(word) >= 4 and word not in GENERATION_STOPWORDS
    }
    if query_terms and len(query_terms.intersection(words)) < min(2, len(query_terms)):
        return False
    trigrams = list(zip(words, words[1:], words[2:]))
    if trigrams and len(set(trigrams)) < len(trigrams) * 0.7:
        return False
    return True


@dataclass
class AssistantSettings:
    backend: BackendName = "hybrid"
    corpus_path: Path = DEFAULT_RETRIEVAL_CORPUS_PATH
    retrieval_index: Path = DEFAULT_RETRIEVAL_INDEX
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
    retrieval_limit: int = 3
    direct_retrieval_threshold: float = 0.50
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    seed: int = 42


@dataclass(frozen=True)
class AssistantResponse:
    text: str
    backend: str
    retrieved: tuple[RetrievedDocument, ...] = ()


class AdvancedAssistant:
    """Tools + trained retrieval + a transformer initialized and trained here."""

    def __init__(self, settings: AssistantSettings | None = None) -> None:
        self.settings = settings or AssistantSettings()
        self.retriever = TrainedRetriever(
            self.settings.corpus_path, self.settings.retrieval_index
        )
        self._transformer: ScratchTransformerBackend | None = None
        self.history: list[dict[str, str]] = []

    @property
    def transformer(self) -> ScratchTransformerBackend:
        if self._transformer is None:
            self._transformer = ScratchTransformerBackend(
                checkpoint_path=self.settings.checkpoint_path,
                tokenizer_path=self.settings.tokenizer_path,
                settings=self.settings.generation,
                seed=self.settings.seed,
            )
        return self._transformer

    def clear_history(self) -> None:
        self.history.clear()

    def _remember(self, query: str, response: str) -> None:
        self.history.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        )
        if len(self.history) > self.settings.max_history_messages:
            self.history = self.history[-self.settings.max_history_messages :]

    def _prompt(self, query: str) -> str:
        parts = ["<bos><system>", self.settings.system_prompt]
        for message in self.history[-self.settings.max_history_messages :]:
            role = "<user>" if message["role"] == "user" else "<assistant>"
            parts.extend([role, message["content"]])
        parts.extend(["<user>", query, "<assistant>"])
        return "\n".join(parts)

    def respond(self, query: str) -> AssistantResponse:
        query = query.strip()
        if not query:
            return AssistantResponse("Please enter a question or request.", "validation")

        tool_result = run_tool(query)
        if tool_result:
            self._remember(query, tool_result.text)
            return AssistantResponse(tool_result.text, f"tool:{tool_result.name}")

        documents = self.retriever.search(query, self.settings.retrieval_limit)
        strong_match = bool(
            documents
            and documents[0].score >= self.settings.direct_retrieval_threshold
        )
        if self.settings.backend == "retrieval" or (
            self.settings.backend == "hybrid" and strong_match
        ):
            text = documents[0].response
            self._remember(query, text)
            return AssistantResponse(text, "retrieval", tuple(documents))

        generated = self.transformer.generate(self._prompt(query))
        grounded = generation_is_grounded(query, generated)
        if (
            self.settings.backend == "hybrid"
            and not grounded
            and documents
            and documents[0].score >= 0.30
        ):
            generated = documents[0].response
            backend = "retrieval-fallback"
        elif self.settings.backend == "hybrid" and not grounded:
            generated = (
                "I do not have a reliable answer for that in my local training data yet."
            )
            backend = "uncertain"
        else:
            backend = "transformer"
        if not generated:
            generated = "I am not confident enough to answer that yet."
        self._remember(query, generated)
        return AssistantResponse(generated, backend, tuple(documents))
