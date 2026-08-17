# SmallLM — From-Scratch Local Language Model

## What this project is

SmallLM is a new decoder-only transformer whose tokenizer and 6.8 million
parameters are trained from random initialization by this repository. It does
not import or wrap another language model and does not use pretrained weights.
It is a compact, educational local assistant: the model, retrieval data,
inference code, training code, and terminal chat interface can all be inspected
and run from this repository.

## How it works

The assistant combines four components:

1. a byte-level BPE tokenizer trained on the project corpus;
2. a six-layer causal transformer trained for instruction responses;
3. a word-and-character TF-IDF retrieval index trained on an expanded human
   instruction corpus;
4. deterministic tools for arithmetic, dates, unit conversion, JSON, and text
   statistics, plus common professional email drafts.

This hybrid design gives precise answers when a close training example exists,
uses the transformer for novel generation, and handles calculations with code
instead of guessing.

For each prompt, SmallLM first checks whether a deterministic tool can answer
it. Otherwise, the retrieval system ranks the locally stored training scenarios.
In the recommended `hybrid` mode, a strong match is returned directly; when no
match is strong enough, the project's transformer generates a response using
the recent chat history. A grounding check rejects disconnected generations and
uses a retrieved fallback or clearly reports uncertainty. The color terminal UI
displays this response pipeline as an ordinary streaming chat experience.

## Project scale

- 11,262 filtered instruction-response pairs
- 10,968 human-written OpenAssistant pairs and 294 original curated pairs
- 25,189 retrieval scenarios, including 13,927 human-written Dolly records
- 8,000-token tokenizer vocabulary
- 6,836,224 trainable transformer parameters
- 6 transformer layers, 8 attention heads, 256 hidden dimensions
- 192-token context window
- 7,056 full training steps over eight epochs
- final validation loss 4.5536 and perplexity 94.97
- no pretrained model or embedding weights
- complete project storage remains below 2 GB, including the local environment

See [DATA_SOURCES.md](DATA_SOURCES.md) for dataset provenance and filtering.

## Start chatting

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once, then
double-click **Start_SmallLM.bat**. That is the only public entry point.

On the first launch, the launcher automatically creates the Python environment
and installs the required packages. Later launches open the chat interface
directly. No SmallLM command-line commands are needed.

### Why Python 3.11 and uv?

Python 3.11 is the version used to build and test this repository. It satisfies
the declared Python requirements of the pinned numerical and machine-learning
packages, including the current NumPy and scikit-learn builds, while providing
well-supported Windows wheels. Newer Python versions may work, but they are not
the tested baseline for the committed checkpoint and scripts.

`uv` keeps this choice isolated from the computer's system Python. The first-run
setup creates `.venv` with Python 3.11 and installs the exact package versions
from `requirements.txt`. If 3.11 is not already installed, `uv` can
[download the requested Python version automatically](https://docs.astral.sh/uv/guides/install-python/).
This makes setup repeatable and avoids changing packages belonging to other
Python projects or the operating system.

No model download occurs during setup because the trained tokenizer, retrieval
index, and SmallLM checkpoint are already committed to this repository.

## Chat interface

The home page has two large choices:

- **Start Chat** opens a spacious, color-coded conversation screen. SmallLM
  displays animated thinking dots while it works, then progressively types its
  answer into a response panel.
- **Settings** opens every user-facing model, conversation, and display control.

The settings screen includes:

- response backend, retrieval threshold, and number of retrieval candidates;
- temperature, top-k sampling, response length, repetition penalty, no-repeat
  phrase size, and random seed;
- system prompt and conversation-memory length;
- response typing speed and minimum thinking-animation time.

Settings are session-only by design. They are never written to disk, so every
new launch restores the documented defaults. The active model settings remain
visible in the upper-right corner of the chat screen. Enter `/settings` while
chatting to change them immediately without leaving the conversation.

Chat commands:

- `/settings` edits and applies all runtime settings;
- `/info` shows model architecture, dataset, and training statistics;
- `/clear` erases the current conversation memory;
- `/help` lists the available chat commands;
- `/back` returns to the home page;
- `/quit` closes SmallLM.

Selecting the `transformer` backend in Settings provides direct transformer
generation. The default `hybrid` backend keeps the grounding and retrieval
fallbacks enabled.

## Training and reproducibility

Dataset preparation downloads OASST1, OASST2, and Databricks Dolly, applies the
documented safety and quality filters, removes duplicates, and writes separate
deterministic transformer-training and retrieval JSONL files.

Training runs on CPU, holds out five percent of examples for validation, masks
prompt tokens so loss focuses on assistant responses, uses AdamW with warmup and
cosine decay, clips gradients, and saves the best validation checkpoint after
each epoch. All random generators use a configurable seed.

Generated reports are saved under `artifacts/` with the trained checkpoint,
tokenizer, and retrieval index. Training, evaluation, and the retained
`main.py` CLI are internal maintenance utilities rather than public usage paths.

## Repository layout

```text
Main_Run_Program.py             color terminal home, settings, and chat UI
Start_SmallLM.bat               public one-click launcher and first-run setup
main.py                         internal maintenance CLI
smalllm/backend.py              transformer architecture and generation
smalllm/assistant.py            tools, retrieval, memory, routing
smalllm/retrieval.py            from-scratch TF-IDF retrieval model
smalllm/tools.py                deterministic utility tools
scripts/prepare_dataset.py      internet dataset filtering and assembly
scripts/build_index.py          retrieval training
scripts/train_transformer.py    tokenizer and transformer training loop
scripts/evaluate.py             multi-scenario smoke evaluation
data/instructions.json          original curated examples
data/instructions_large.jsonl   11,262-pair training corpus
data/knowledge_large.jsonl      25,189-scenario retrieval corpus
artifacts/tokenizer.json        trained BPE tokenizer
artifacts/retrieval_index.joblib trained sparse retrieval weights
artifacts/smalllm_checkpoint.pt trained transformer weights
artifacts/training_report.json  reproducible metrics and configuration
```

## Limitations

SmallLM is genuinely trained from scratch, but 6.8 million parameters and
11,262 transformer-training pairs are tiny compared with commercial LLMs. The
raw `transformer` backend can still produce incorrect, repetitive, fragmented,
or shallow responses. The recommended `hybrid` backend is more reliable because
it combines 25,189 retrieval scenarios, deterministic tools, and a grounding
gate, but retrieval quality still depends on corpus coverage. Its 192-token
context is short. Treat it as an educational and locally inspectable model, not
an authoritative source. Verify high-stakes medical, legal, financial,
security, and factual outputs independently.
