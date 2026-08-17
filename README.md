# Basic Language Model

This repository contains a small character-level n-gram language model written
with the Python standard library. It learns character transition counts from a
text corpus, backs off to shorter contexts when needed, and samples new text.

It is intentionally compact and inspectable. No GPU, API key, download, or
third-party package is required.

## Quick start

Python 3.10 or newer is recommended.

```powershell
python main.py generate --prompt "The model" --length 300 --seed 42
```

The repository already includes `model.json`, trained on
`data/training.txt`, so generation works immediately.

## Train the model

```powershell
python main.py train --data data/training.txt --model model.json --order 5
```

Replace `data/training.txt` with any UTF-8 plain-text file to train on your own
data. More consistent and representative text generally produces more
consistent output. Only use data that you have permission to use.

## Generate text

```powershell
python main.py generate `
  --model model.json `
  --prompt "Language is" `
  --length 500 `
  --temperature 0.8 `
  --seed 7
```

- `--prompt` supplies the opening text.
- `--length` is the number of new characters to produce.
- `--temperature 0` always picks the most common next character. Values around
  `0.7` to `1.0` add variety.
- `--seed` makes sampled output reproducible.

Inspect the trained model metadata with:

```powershell
python main.py info --model model.json
```

## Run tests

```powershell
python -m unittest -v
```

## How it works

For every character in the corpus, the trainer records what followed contexts
from zero up to five characters long. During generation, the model uses the
longest matching context. If that context was not present in training, it tries
progressively shorter suffixes until it finds learned counts. Temperature is
applied to those counts before the next character is sampled.

This model is educational rather than production-grade. Natural next steps
include word tokenization, evaluation on a held-out dataset, neural embeddings,
and a transformer implementation.
