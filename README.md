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

## Install

Python 3.11 is recommended. With [uv](https://docs.astral.sh/uv/) installed:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

### Why Python 3.11 and uv?

Python 3.11 is the version used to build and test this repository. It satisfies
the declared Python requirements of the pinned numerical and machine-learning
packages, including the current NumPy and scikit-learn builds, while providing
well-supported Windows wheels. Newer Python versions may work, but they are not
the tested baseline for the committed checkpoint and scripts.

`uv` keeps this choice isolated from the computer's system Python. The setup
script asks it to create `.venv` with Python 3.11 and install the exact package
versions from `requirements.txt`. If 3.11 is not already installed, `uv` can
[download the requested Python version automatically](https://docs.astral.sh/uv/guides/install-python/).
This makes setup repeatable and avoids changing packages belonging to other
Python projects or the operating system.

The environment contains PyTorch and data-processing libraries only. No model
download occurs because the trained tokenizer, retrieval index, and SmallLM
checkpoint are committed to this repository.

## Ask questions

```powershell
.venv\Scripts\python.exe main.py ask --prompt "Hello"
.venv\Scripts\python.exe main.py ask --prompt "How do I debug an HTTP 500 error?"
.venv\Scripts\python.exe main.py ask --prompt "Convert 25 celsius to fahrenheit"
```

Show which component answered and the retrieval score:

```powershell
.venv\Scripts\python.exe main.py ask `
  --prompt "Explain photosynthesis simply" `
  --show-score
```

Choose a backend explicitly:

```powershell
# Recommended: retrieval for strong matches, transformer otherwise
.venv\Scripts\python.exe main.py ask --backend hybrid --prompt "Your question"

# Always generate with the new transformer
.venv\Scripts\python.exe main.py ask --backend transformer --prompt "Your question"

# Only retrieve a trained response
.venv\Scripts\python.exe main.py ask --backend retrieval --prompt "Your question"
```

## Color terminal interface

After running `setup.ps1`, double-click **Start_SmallLM.bat** on Windows. This
launcher works even when Windows has no usable `.py` file association. You can
also double-click **Main_Run_Program.py** when `.py` files are associated with
Python, or launch it from a terminal:

```powershell
.venv\Scripts\python.exe Main_Run_Program.py
```

The home page has two large choices:

- **Start Chat** opens a spacious, color-coded conversation screen. SmallLM
  displays animated thinking dots while it works, then progressively types its
  answer into a response panel.
- **Settings** changes the response backend, temperature, top-k sampling,
  response length, repetition controls, retrieval threshold, and typing speed.

Settings are session-only by design. They are never written to disk, so every
new launch restores the documented defaults. The active model settings remain
visible in the upper-right corner of the chat screen.

While chatting, use `/clear` to erase conversation memory, `/help` to list
commands, `/back` to return to the home page, or `/quit` to close the program.
The original `main.py` command-line interface remains available for scripting.

## Interactive chat

```powershell
.venv\Scripts\python.exe main.py chat --show-backend
```

The chat loop keeps the recent conversation in the transformer's prompt. The
hybrid backend rejects topically disconnected generations and falls back to a
retrieved human response or an explicit uncertainty. Type `quit` or `exit` to
stop.

## Direct transformer generation

```powershell
.venv\Scripts\python.exe main.py generate `
  --prompt "<bos><system>`nYou are helpful.`n<user>`nExplain gravity.`n<assistant>`n" `
  --max-new-tokens 100 `
  --temperature 0.7 `
  --top-k 40 `
  --seed 42
```

## Rebuild the data and train from scratch

Dataset preparation downloads OASST1, OASST2, and Databricks Dolly, applies the
documented safety and quality filters, removes duplicates, and writes separate
deterministic transformer-training and retrieval JSONL files:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-data.txt
.venv\Scripts\python.exe main.py prepare-data
```

Train the retrieval index, tokenizer, and transformer:

```powershell
.venv\Scripts\python.exe main.py train `
  --epochs 8 `
  --batch-size 12 `
  --rebuild-tokenizer
```

Training runs on CPU, holds out five percent of examples for validation, masks
prompt tokens so loss focuses on assistant responses, uses AdamW with warmup and
cosine decay, clips gradients, and saves the best validation checkpoint after
each epoch. All random generators use a configurable seed. Add `--resume` to
continue a checkpoint on the same checksummed corpus with a fresh optimizer.

Advanced architecture options are available directly through
`scripts/train_transformer.py`:

```powershell
.venv\Scripts\python.exe scripts\train_transformer.py --help
```

## Inspect and evaluate

```powershell
.venv\Scripts\python.exe main.py info
.venv\Scripts\python.exe scripts\evaluate.py --backend retrieval
.venv\Scripts\python.exe scripts\evaluate.py --backend hybrid --limit 5
.venv\Scripts\python.exe -m unittest -v
```

Generated reports are saved under `artifacts/` with the trained checkpoint,
tokenizer, and retrieval index.

## Repository layout

```text
Main_Run_Program.py             color terminal home, settings, and chat UI
Start_SmallLM.bat               reliable one-click Windows launcher
main.py                         command-line interface
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
