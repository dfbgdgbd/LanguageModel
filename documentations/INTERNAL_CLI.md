# Internal CLI and scripts

## Scope warning

The commands in this document are for maintainers. The supported public user
workflow is `Start_SmallLM.bat` plus the chat interface. Keep these commands out
of the public usage section of the root README.

Run examples from the repository root with the project interpreter.

## `main.py` command map

### Prepare data

```powershell
.venv\Scripts\python.exe main.py prepare-data
```

Optional limits accept nonnegative record counts; `0` means no limit after
filtering:

```powershell
.venv\Scripts\python.exe main.py prepare-data `
  --oasst1-limit 1000 `
  --oasst2-limit 1000 `
  --dolly-limit 2000
```

This delegates to `scripts/prepare_dataset.py`. Install
`requirements-data.txt` first. Upstream downloads require network access.

### Train

```powershell
.venv\Scripts\python.exe main.py train --epochs 8 --batch-size 12
```

The wrapper first rebuilds the retrieval index, then starts transformer
training. Relevant flags:

- `--max-steps N` caps new transformer steps;
- `--seed N` controls data shuffle, initialization, and sampling metadata;
- `--rebuild-tokenizer` replaces the tokenizer before training;
- `--resume` loads model weights from the existing compatible checkpoint.

Do not combine `--resume` with `--rebuild-tokenizer`. Resume uses a fresh
optimizer and scheduler; it is weight continuation, not exact optimizer-state
continuation.

### Single internal query

```powershell
.venv\Scripts\python.exe main.py ask --prompt "Hello" --show-score
```

Options include `--backend`, `--threshold`, `--max-new-tokens`, `--temperature`,
`--top-k`, and `--seed`. This is useful for automation and diagnostics, not the
public interface.

### Plain internal chat

```powershell
.venv\Scripts\python.exe main.py chat --show-backend
```

This is intentionally minimal and does not replace testing the Rich interface.

### Direct transformer generation

```powershell
.venv\Scripts\python.exe main.py generate `
  --prompt "<bos><system>`nYou are helpful.`n<user>`nHello`n<assistant>`n" `
  --temperature 0.75 `
  --top-k 40 `
  --max-new-tokens 64
```

This bypasses tools, retrieval, conversation orchestration, and the hybrid
grounding fallback.

### Model information

```powershell
.venv\Scripts\python.exe main.py info
```

Public users get the corresponding information through `/info` in chat.

## Direct scripts

Use direct scripts when the wrapper does not expose the required maintenance
option.

### Dataset builder

```powershell
.venv\Scripts\python.exe scripts\prepare_dataset.py --help
```

Supports custom curated input and custom training/retrieval output paths in
addition to source limits.

### Retrieval builder

```powershell
.venv\Scripts\python.exe scripts\build_index.py `
  --corpus data\knowledge_large.jsonl `
  --output artifacts\retrieval_index.joblib
```

### Transformer trainer

```powershell
.venv\Scripts\python.exe scripts\train_transformer.py --help
```

Direct options cover corpus/artifact paths, vocabulary, context length, model
width, layer/head/feed-forward counts, dropout, optimizer settings, warmup,
threads, maximum steps, tokenizer rebuild, and resume. The direct script
defaults to 3 epochs; the `main.py train` wrapper defaults to 8.

### Evaluation

```powershell
.venv\Scripts\python.exe scripts\evaluate.py `
  --backend hybrid `
  --output artifacts\evaluation_report.json
```

`--limit` reduces the built-in 20-scenario set. The report records query,
response, answering backend, elapsed seconds, and top retrieval score.

## Exit behavior

`main.py` converts common filesystem, data, type, JSON, and subprocess failures
into an error message and exit code `1`. Argument parser errors use argparse's
normal nonzero exit. Direct scripts may raise uncaught exceptions; retain the
traceback for diagnosis.
