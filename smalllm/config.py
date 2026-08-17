"""Shared defaults and artifact paths for SmallLM."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "instructions_large.jsonl"
DEFAULT_RETRIEVAL_CORPUS_PATH = PROJECT_ROOT / "data" / "knowledge_large.jsonl"
DEFAULT_RETRIEVAL_INDEX = PROJECT_ROOT / "artifacts" / "retrieval_index.joblib"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "artifacts" / "tokenizer.json"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "smalllm_checkpoint.pt"
DEFAULT_TRAINING_REPORT = PROJECT_ROOT / "artifacts" / "training_report.json"
DEFAULT_MAX_HISTORY_MESSAGES = 8
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<system>",
    "<user>",
    "<assistant>",
]
DEFAULT_SYSTEM_PROMPT = (
    "You are SmallLM, a helpful local assistant. Follow the request, be concise, "
    "and say when you are uncertain."
)
