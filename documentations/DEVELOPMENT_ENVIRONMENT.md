# Development environment

## Supported baseline

Development and tests use Python 3.11 in `.venv`. The pinned NumPy and
scikit-learn versions declare Python 3.11 or newer, while the remaining runtime
packages also support 3.11. Newer Python releases may work but are not the
validated baseline.

`uv` is used because it can locate or download the requested interpreter,
create an isolated environment, and install exact package versions without
modifying system Python.

## Create or repair the runtime environment

Public users should double-click `Start_SmallLM.bat`; it performs these checks
automatically. A maintainer can run the internal setup script directly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

The script:

1. verifies that `uv` is on `PATH`;
2. creates `.venv` with Python 3.11 if its interpreter is absent;
3. installs `requirements.txt` into that exact environment.

## Runtime dependencies

`requirements.txt` pins:

- CPU-only PyTorch for the transformer;
- NumPy for numerical work and retrieval scoring;
- Hugging Face `tokenizers` for byte-level BPE;
- scikit-learn for TF-IDF vectorizers and sparse matrices;
- joblib for retrieval artifact serialization;
- tqdm for training progress;
- Rich for terminal layout, color, status animation, and live response panels.

The PyTorch CPU index is configured at the top of the requirements file. Do not
remove it unless GPU packaging is deliberately introduced and tested.

## Data-development dependency

Dataset rebuilding is intentionally separate because Hugging Face `datasets`
and its transitive packages add substantial weight:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-data.txt
```

Do not move data-only dependencies into `requirements.txt` without reconsidering
first-run time and the 2 GiB project limit.

## Running Python during development

Use the interpreter explicitly so commands do not accidentally use the broken
or differently versioned global `.py` association:

```powershell
.venv\Scripts\python.exe -m unittest -v
.venv\Scripts\python.exe main.py info
```

The second command is an internal maintenance example. Public documentation
must continue to direct users to `Start_SmallLM.bat`.

## Updating dependencies

1. Confirm the candidate package supports Python 3.11 and Windows.
2. Update one logical dependency group at a time.
3. Recreate or update `.venv` through `setup.ps1`.
4. Run the complete unit suite and launch/chat smoke tests.
5. Check artifact loading; joblib and PyTorch serialization can be
   version-sensitive.
6. Recheck total project size.
7. Commit the requirements change with any compatibility documentation.

Do not commit `.venv`, caches, raw downloads, temporary checkpoints, or
`__pycache__`; `.gitignore` excludes them.

## Line endings and cross-platform hashes

`.gitattributes` enforces LF for source/data text and CRLF for PowerShell.
Binary `.pt` and `.joblib` files are marked binary. `corpus_sha256()` normalizes
CRLF and CR to LF before hashing, so checkout line-ending differences do not
invalidate trained artifacts.

When adding a new generated text format that participates in hashes, add an
explicit `.gitattributes` rule and use canonical hashing.
