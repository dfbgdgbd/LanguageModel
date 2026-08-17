# SmallLM developer documentation

This folder is the maintenance handbook for SmallLM. It is written for future
developers who need to understand, test, modify, retrain, or release the
project without reverse-engineering every file first.

The public user experience is intentionally simple: install `uv`, then
double-click `Start_SmallLM.bat`. The internal CLI and scripts remain available
for maintainers and are documented here, not in the public usage section of the
root README.

## Documentation map

| Document | Use it when you need to... |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | understand startup, chat state, response routing, and module ownership |
| [Chat interface](CHAT_INTERFACE.md) | change menus, commands, settings, animation, or terminal rendering |
| [Model and inference](MODEL_AND_INFERENCE.md) | modify the tokenizer, transformer, retrieval scoring, prompts, or grounding gate |
| [Development environment](DEVELOPMENT_ENVIRONMENT.md) | create a development environment or update dependencies safely |
| [Internal CLI](INTERNAL_CLI.md) | use the retained maintenance commands and scripts |
| [Data and training](DATA_AND_TRAINING.md) | rebuild datasets, retrain artifacts, or review filtering and licensing |
| [Testing and evaluation](TESTING_AND_EVALUATION.md) | run regression tests, smoke checks, and qualitative evaluation |
| [Artifacts and versioning](ARTIFACTS_AND_VERSIONING.md) | inspect or replace the tokenizer, checkpoint, index, and reports |
| [Maintenance and releases](MAINTENANCE_AND_RELEASES.md) | prepare a safe change, review it, and publish it |
| [Troubleshooting](TROUBLESHOOTING.md) | diagnose startup, artifact, model-quality, or terminal problems |

## Source-of-truth order

When documentation and code disagree, use this order and then correct the
documentation in the same change:

1. executable validation in `test_main.py`;
2. implementation in `Main_Run_Program.py`, `smalllm/`, and `scripts/`;
3. committed metadata in `artifacts/training_report.json` and the artifact
   payloads;
4. this handbook;
5. the public `README.md`.

Do not silently change an artifact format, routing threshold, dataset schema,
or public setting without updating its relevant documentation and tests.

## Current baseline

- Python: 3.11
- Public entry point: `Start_SmallLM.bat`
- Model: 6,836,224-parameter decoder-only transformer
- Tokenizer: 8,000-token byte-level BPE
- Context: 192 tokens
- Transformer training corpus: 11,262 records
- Retrieval corpus: 25,189 records
- Retrieval artifact format: version 3
- Checkpoint artifact format: version 1
- Test runner: standard-library `unittest`
- Storage requirement: entire working project must remain below 2 GiB
- Pretrained model or tokenizer weights: none

These values describe the committed baseline. After retraining, update this
index, the root README, model information shown by `/info`, and any affected
artifact documentation together.

## Fast maintainer checklist

Before changing code:

1. read the document for the subsystem you are modifying;
2. start from a clean `main` branch and create a feature branch;
3. preserve the public chat-only boundary unless the product decision changes;
4. keep generated artifacts and their source corpora synchronized.

Before publishing:

1. run the full unit suite;
2. smoke-test `Start_SmallLM.bat`, Settings, `/settings`, `/info`, and a real
   prompt;
3. run `git diff --check`;
4. confirm the full project remains below 2 GiB;
5. inspect the staged diff and update documentation for any changed contract.
