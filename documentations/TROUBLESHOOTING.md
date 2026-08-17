# Troubleshooting

## Launcher says `uv` is missing

The first run needs `uv` to create Python 3.11 and install dependencies. Install
it from the official Astral instructions, close/reopen the terminal if `PATH`
changed, then double-click `Start_SmallLM.bat` again.

Diagnostic:

```powershell
uv --version
```

## A `.py` file does not open correctly

Windows file associations can point to a missing Python Launcher runtime. Do
not repair the whole system association just for SmallLM. Use
`Start_SmallLM.bat`, which directly selects `.venv\Scripts\python.exe`.

## Missing Python package or Rich import error

The batch launcher imports all runtime dependencies before starting. If one is
missing, it reruns `setup.ps1`. For manual repair:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

If repair repeatedly fails, inspect network/proxy access, available disk, the
PyTorch CPU package index, and the first error above the final failure.

## Retrieval index does not match corpus

Symptoms include:

- `retrieval index and corpus have different record counts`;
- `retrieval index does not match the corpus`;
- `unsupported retrieval index version`.

Do not bypass validation. Confirm the intended `knowledge_large.jsonl`, then
rebuild `artifacts/retrieval_index.joblib` with `scripts/build_index.py`. If the
corpus change was accidental, restore the matching committed pair instead.

## Checkpoint fails to load

Common errors:

- `unsupported checkpoint format`: loader and artifact format versions differ;
- tokenizer/checkpoint vocabulary mismatch: tokenizer and checkpoint were not
  trained as a compatible pair;
- missing checkpoint/tokenizer: committed artifacts are absent or path defaults
  were changed.

Restore matching committed artifacts or perform a deliberate tokenizer and
transformer rebuild. Do not edit stored config values by hand.

## First response is slow

Chat startup loads the 25,189-record retrieval corpus and its sparse index. A
prompt routed to generation then lazily loads the transformer checkpoint. Later
retrieval responses should avoid transformer load. CPU transformer generation
is inherently slower than retrieval or tools.

Use `/info` to confirm the model baseline and inspect the response footer to see
which backend answered.

## Hybrid returns an unrelated retrieved answer

1. note the query, backend footer, and score;
2. reproduce with the internal `ask --show-score` diagnostic;
3. inspect the top retrieval records and normalized terms;
4. test a higher retrieval threshold temporarily in chat;
5. add an evaluation and regression case before changing global scoring.

Do not tune only for one prompt; threshold and term changes affect all domains.

## Hybrid reports uncertainty too often

The transformer generation may be failing the grounding gate and the best
retrieval score may be below `0.30`. Inspect raw transformer output through the
internal transformer backend, add targeted training/retrieval coverage if
appropriate, and evaluate rather than simply lowering safety thresholds.

## Transformer output is repetitive or fragmented

This is expected risk for a 6.8M-parameter model trained on limited data. Try:

- lower temperature;
- smaller top-k;
- higher repetition penalty;
- a larger no-repeat phrase size;
- the default hybrid backend;
- improved relevant training data followed by evaluation/retraining.

Settings can reduce symptoms but cannot supply missing model capacity or
knowledge.

## Settings do not persist

This is intentional. `RuntimeSettings` is recreated on every process start and
there is no settings file. Persistent settings require an explicit product and
privacy decision, serialization format, migration policy, UI indication, and
new tests.

## Settings changed but conversation disappeared

`/settings` preserves the newest history within the new memory limit. Starting
a chat from the home screen creates a new assistant. Reducing Conversation
memory intentionally drops older messages.

## Garbled characters or terminal encoding error

Keep app-authored labels ASCII-safe and let Rich choose compatible box glyphs.
Test redirected output in the classic Windows console. Non-ASCII comparison
signs or decorative bullets can fail under CP1252 even if they work in Windows
Terminal.

## Settings table is unreadable at narrow width

The current layout is designed for 80 columns with setting descriptions grouped
under labels. If new fields or longer copy are added, test 80-column output and
adjust column ratios/wrapping rather than allowing Rich to truncate a column to
replacement glyphs.

## Tests fail on project size

Find large paths before deleting anything:

```powershell
Get-ChildItem -LiteralPath . -File -Recurse |
  Sort-Object Length -Descending |
  Select-Object -First 30 FullName, Length
```

Likely causes are raw dataset caches, temporary checkpoints, a duplicate virtual
environment, or new binary history. Preserve required committed artifacts. Do
not delete or rewrite Git history without an explicit reviewed plan.

## Training resume is rejected

Resume requires:

- an existing format-version-1 checkpoint;
- the exact same architecture arguments;
- the same canonical transformer corpus hash;
- no simultaneous tokenizer rebuild.

If any invariant changed, start a new compatible training run rather than
forcing resume.
