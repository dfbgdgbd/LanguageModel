# Testing and evaluation

## Required regression suite

Run from the repository root:

```powershell
.venv\Scripts\python.exe -m unittest -v
```

The current suite contains 25 tests and takes roughly 20–25 seconds on the
reference Windows CPU environment. It intentionally loads the real tokenizer,
checkpoint, corpora, and retrieval index.

## Test groups

### `ToolTests`

Validates arithmetic, conversions, text statistics, JSON behavior, email and
guided responses, and resistance to arbitrary Python execution. When adding a
tool, include a positive case and a near-miss or unsafe case.

### `CorpusAndRetrievalTests`

Validates exact transformer corpus counts/sources, minimum retrieval corpus
coverage, expected search behavior, tool/retrieval routing, conversation memory,
and the grounding heuristic.

Exact count assertions are artifact-integrity checks. Update them only when a
deliberate, reviewed corpus rebuild changes the baseline.

### `FromScratchModelTests`

Validates tokenizer size and required tokens, architecture and parameter count,
fixed-seed generation, absence of leaked control tokens, and report metadata
showing random initialization with no pretrained weights.

### `ProjectTests`

Validates:

- all runtime settings are temporary and validated;
- every `RuntimeSettings` field is exposed by the UI;
- translation into assistant/generation settings;
- in-chat model information;
- chat-only public README usage;
- the retained internal CLI smoke path;
- the complete project remains below 2 GiB.

## Focused tests during development

The suite uses standard `unittest`, so a class or method can be targeted:

```powershell
.venv\Scripts\python.exe -m unittest -v test_main.ProjectTests
.venv\Scripts\python.exe -m unittest -v `
  test_main.ProjectTests.test_terminal_settings_validate_and_translate
```

Always run the complete suite before publishing even when focused tests pass.

## Terminal smoke tests

Automated unit tests do not prove that a Rich layout is readable. At minimum,
manually or with redirected input verify:

1. home screen and quit;
2. home Settings, edit, reset, and back;
3. Start Chat and a known prompt such as `Hello`;
4. `/settings`, change a value, apply, and continue the same conversation;
5. `/info` and `/help`;
6. `/clear`, `/back`, and `/quit`;
7. an 80-column classic console and Windows Terminal;
8. progressive output and thinking animation at zero and nonzero delays.

A non-interactive Windows example is:

```powershell
@('1', 'Hello', '/quit') | & .\Start_SmallLM.bat
```

This verifies execution and rendering output, but not colors or animation
smoothness.

## Qualitative evaluation

`scripts/evaluate.py` contains 20 scenarios spanning explanations, programming,
writing, comparison, study planning, debugging, arithmetic, conversion, JSON,
date, security, and brainstorming.

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --backend retrieval
.venv\Scripts\python.exe scripts\evaluate.py --backend hybrid --limit 20
```

Review `artifacts/evaluation_report.json`; do not treat generation completion as
quality success. Inspect relevance, factuality, repetition, backend choice,
latency, and retrieval scores.

Transformer evaluation can be slow on CPU. Use a small `--limit` during rapid
iteration, then run a broader check before an artifact release.

## Acceptance gates by change type

| Change | Minimum validation |
| --- | --- |
| README/docs only | link/path review, `git diff --check`, relevant documentation test |
| Terminal copy/layout | full tests plus 80-column and Windows Terminal smoke tests |
| Setting or chat command | boundary/unit tests, full suite, live application smoke test |
| Tool handler | positive, near-miss, and unsafe-input tests |
| Retrieval logic/index | corpus/index integrity, retrieval cases, evaluation report review |
| Transformer/generation | fixed-seed tests, full suite, qualitative generation review |
| Corpus or artifact | counts, licenses, checksums, full suite, evaluation, storage check |
| Dependency | clean setup, imports, full suite, launcher smoke, artifact compatibility |

## Storage check

The test suite checks the entire project tree, including `.venv`, against
2 GiB. A direct diagnostic is:

```powershell
$bytes = (Get-ChildItem -LiteralPath . -File -Recurse |
  Measure-Object -Property Length -Sum).Sum
[math]::Round($bytes / 1GB, 3)
```

The current environment is approximately 0.961 GiB.

## Continuous integration status

There is currently no committed GitHub Actions workflow. Until CI is added, the
developer publishing a change owns the local validation gates. If CI is added,
avoid downloading unnecessary data or retraining; use committed artifacts and
the standard unit suite.
