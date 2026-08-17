# Maintenance and releases

## Change discipline

SmallLM combines source, large generated artifacts, licensed data, and public UI
behavior. Treat changes as one of these classes before editing:

- public UI or documentation;
- assistant routing or tools;
- retrieval/data;
- transformer/tokenizer/training;
- dependencies/environment;
- artifact-only refresh.

Read the relevant handbook pages and identify every coupled file first.

## Branch and review workflow

1. Synchronize a clean local `main` with `origin/main`.
2. Create a focused feature branch such as `agent/short-description`.
3. Make scoped changes; preserve unrelated worktree changes.
4. Run proportionate focused checks during development.
5. Run the complete acceptance gates before committing.
6. Stage explicit paths and inspect `git diff --cached` plus
   `git diff --cached --check`.
7. Use a terse commit that describes the full change.
8. Push with upstream tracking.
9. Open a pull request describing what, why, impact, and validation.
10. Merge only when the PR is mergeable and the head SHA is still expected.
11. Synchronize local `main` and verify remote content after merge.

The repository currently uses squash merges for agent-produced feature PRs.

## Public interface invariant

The only documented end-user entry point is `Start_SmallLM.bat`. `main.py` and
`scripts/` are retained for maintainers. A change touching public usage must
keep these aligned:

- root README;
- launcher behavior;
- chat help/header;
- this handbook;
- `test_readme_has_chat_only_public_usage`.

## Documentation synchronization matrix

| Change | Update at minimum |
| --- | --- |
| UI setting | `CHAT_INTERFACE.md`, root README summary, setting tests |
| Chat command | `CHAT_INTERFACE.md`, root README command list, help rendering |
| Routing/threshold | `ARCHITECTURE.md`, `MODEL_AND_INFERENCE.md`, routing tests |
| Model config/metrics | root README, docs index baseline, model/artifact docs, `/info` tests |
| Dataset/count/license | `DATA_SOURCES.md`, `DATA_AND_TRAINING.md`, `NOTICE`/licenses as required, count tests |
| Artifact schema | `ARTIFACTS_AND_VERSIONING.md`, loader, version, tests |
| Dependency/setup | root README, `DEVELOPMENT_ENVIRONMENT.md`, launcher/setup scripts |
| Internal command | `INTERNAL_CLI.md` |

## Adding functionality

Prefer placing behavior in the narrowest responsible module:

- UI-only behavior in `Main_Run_Program.py`;
- orchestration in `smalllm/assistant.py`;
- deterministic exact behavior in `smalllm/tools.py`;
- retrieval behavior in `smalllm/retrieval.py`;
- neural model/generation behavior in `smalllm/backend.py`;
- maintenance workflows in `scripts/` and `main.py`.

Avoid placing model logic inside Rich rendering methods. Keep core behavior
callable from tests without an interactive terminal.

## Dependency releases

Because artifacts use PyTorch and joblib serialization, dependency updates are
not routine text-only changes. Verify that both binary artifacts load, run the
full suite, run retrieval/evaluation smoke tests, and test a clean setup. Record
any artifact rebuild caused by compatibility.

## Data and license review

For new or modified data:

1. verify redistribution and model-training permissions;
2. retain source IDs and license metadata;
3. update provenance, notices, and license files;
4. review sensitive-data filters and sample output manually;
5. keep transformer-training and retrieval-only roles explicit;
6. consider share-alike obligations for derived datasets/artifacts.

## Release gate

A release or merged feature is ready only when:

- intended source and documentation are committed;
- no secrets, raw downloads, cache, `.venv`, or temporary checkpoint is staged;
- all tests pass;
- relevant UI and model smoke tests pass;
- artifacts load and match their source corpora;
- metrics/counts in documentation match committed reports;
- project storage remains below 2 GiB;
- public usage still points to the one-click chat launcher;
- GitHub `main` is verified after merge.

## Known technical debt

- There is no automated CI workflow.
- The model is CPU-only and has a short 192-token context.
- Upstream dataset revisions are not pinned in the preparation script.
- The terminal UI and internal CLI live in large single files and may benefit
  from future decomposition, provided behavior remains tested.
- Resume training does not restore optimizer/scheduler/RNG state.
- The grounding gate is heuristic, not a factuality verifier.
- Binary artifacts are committed directly, so repeated rebuilds grow Git
  history.

Track deliberate improvements to these items with tests and migration notes.
