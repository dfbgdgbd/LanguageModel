"""A word-and-character retrieval index trained entirely from project data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .config import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_RETRIEVAL_CORPUS_PATH,
    DEFAULT_RETRIEVAL_INDEX,
)


@dataclass(frozen=True)
class RetrievedDocument:
    prompt: str
    response: str
    score: float
    source: str
    record_id: str


QUERY_EXPANSIONS = {
    "duplicate": "unique distinct deduplicate set",
    "duplicates": "unique distinct deduplicate set",
    "reschedule": "postpone rearrange schedule move",
    "summarize": "summary key points",
    "compare": "difference differences versus",
}
DOMAIN_TERMS = {
    "python",
    "javascript",
    "typescript",
    "java",
    "sql",
    "mongodb",
    "excel",
    "git",
    "linux",
    "windows",
    "docker",
    "rust",
    "html",
    "css",
    "json",
}
RETRIEVAL_STOPWORDS = {
    "about",
    "answer",
    "beginner",
    "clear",
    "could",
    "create",
    "explain",
    "first",
    "function",
    "give",
    "help",
    "into",
    "please",
    "short",
    "should",
    "simple",
    "suggest",
    "terms",
    "that",
    "this",
    "tiny",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "write",
}


def normalized_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for word in re.findall(r"[a-z][a-z0-9+#.-]+", text.casefold()):
        if len(word) < 3 or word in RETRIEVAL_STOPWORDS:
            continue
        if len(word) > 5 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 4 and word.endswith("ies"):
            word = f"{word[:-3]}y"
        elif len(word) > 4 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        terms.add(word)
    return terms


def expand_query(query: str) -> str:
    words = set(re.findall(r"[a-z0-9+#.-]+", query.casefold()))
    additions = [value for key, value in QUERY_EXPANSIONS.items() if key in words]
    return " ".join([query, *additions])


def load_corpus(path: str | Path = DEFAULT_CORPUS_PATH) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not record.get("prompt") or not record.get("response"):
                raise ValueError(f"invalid corpus record on line {line_number}")
            records.append(record)
    if not records:
        raise ValueError("retrieval corpus is empty")
    return records


def train_retrieval_index(
    corpus_path: str | Path = DEFAULT_RETRIEVAL_CORPUS_PATH,
    output_path: str | Path = DEFAULT_RETRIEVAL_INDEX,
) -> tuple[int, int]:
    """Fit a hybrid TF-IDF index without any pretrained weights."""
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    records = load_corpus(corpus_path)
    prompts = [str(record["prompt"]) for record in records]
    responses = [str(record["response"]) for record in records]
    vectorizer = FeatureUnion(
        [
            (
                "words",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=60_000,
                    sublinear_tf=True,
                    norm="l2",
                    dtype=np.float32,
                ),
            ),
            (
                "characters",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=40_000,
                    sublinear_tf=True,
                    norm="l2",
                    dtype=np.float32,
                ),
            ),
        ],
        transformer_weights={"words": 1.0, "characters": 0.45},
    )
    prompt_matrix = vectorizer.fit_transform(prompts)
    response_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    response_matrix = response_vectorizer.fit_transform(responses)
    artifact = {
        "version": 3,
        "prompt_vectorizer": vectorizer,
        "response_vectorizer": response_vectorizer,
        "prompt_matrix": prompt_matrix,
        "response_matrix": response_matrix,
        "record_count": len(records),
        "corpus_sha256": hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination, compress=3)
    return len(records), destination.stat().st_size


class TrainedRetriever:
    def __init__(
        self,
        corpus_path: str | Path = DEFAULT_RETRIEVAL_CORPUS_PATH,
        index_path: str | Path = DEFAULT_RETRIEVAL_INDEX,
    ) -> None:
        import joblib

        self.records = load_corpus(corpus_path)
        artifact = joblib.load(index_path)
        if artifact.get("version") != 3:
            raise ValueError("unsupported retrieval index version")
        if int(artifact["record_count"]) != len(self.records):
            raise ValueError("retrieval index and corpus have different record counts")
        checksum = hashlib.sha256(Path(corpus_path).read_bytes()).hexdigest()
        if artifact["corpus_sha256"] != checksum:
            raise ValueError("retrieval index does not match the corpus")
        self.prompt_vectorizer = artifact["prompt_vectorizer"]
        self.response_vectorizer = artifact["response_vectorizer"]
        self.prompt_matrix = artifact["prompt_matrix"]
        self.response_matrix = artifact["response_matrix"]
        self.searchable_text = [
            f"{record['prompt']}\n{record['response']}".casefold()
            for record in self.records
        ]
        self.searchable_terms = [normalized_terms(text) for text in self.searchable_text]

    def search(self, query: str, limit: int = 3) -> list[RetrievedDocument]:
        import numpy as np

        if limit < 1:
            return []
        expanded_query = expand_query(query)
        prompt_query = self.prompt_vectorizer.transform([expanded_query])
        response_query = self.response_vectorizer.transform([expanded_query])
        prompt_scores = (self.prompt_matrix @ prompt_query.T).toarray().ravel()
        response_scores = (self.response_matrix @ response_query.T).toarray().ravel()
        scores = prompt_scores + response_scores
        query_terms = normalized_terms(expanded_query)
        if query_terms:
            coverage = np.fromiter(
                (
                    len(query_terms.intersection(terms)) / len(query_terms)
                    for terms in self.searchable_terms
                ),
                dtype=float,
                count=len(self.searchable_terms),
            )
            scores *= 0.35 + 0.65 * coverage
        query_words = set(re.findall(r"[a-z0-9+#.-]+", query.casefold()))
        for domain in DOMAIN_TERMS.intersection(query_words):
            matches = np.fromiter(
                (domain in text for text in self.searchable_text),
                dtype=float,
                count=len(self.searchable_text),
            )
            scores += matches * 0.2
        indices = np.argsort(scores)[::-1][:limit]
        return [
            RetrievedDocument(
                prompt=str(self.records[index]["prompt"]),
                response=str(self.records[index]["response"]),
                score=float(scores[index]),
                source=str(self.records[index].get("source", "unknown")),
                record_id=str(self.records[index].get("id", index)),
            )
            for index in indices
        ]
