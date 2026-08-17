# SmallLM Query Assistant

SmallLM is a dependency-free educational language-model project. It is trained
on a varied instruction dataset and provides two modes:

- a TF-IDF query-response model for grounded answers; and
- a character n-gram generator for experimental text continuation.

The included `model.json` is already trained, so it works immediately with
Python 3.10 or newer. It requires no GPU, API key, download, or third-party
package.

## Ask a question

```powershell
python main.py ask --prompt "Hello"
python main.py ask --prompt "How do plants turn light into energy?"
python main.py ask --prompt "What is 19 * (4 + 2)?"
```

To inspect the matched topic and confidence score:

```powershell
python main.py ask --prompt "How do I debug a Python script?" --show-score
```

SmallLM returns an honest unknown response when a query does not match its
training data strongly enough. The `--threshold` option controls the minimum
confidence, with `0.24` as the default.

## Interactive chat

```powershell
python main.py chat
```

Enter questions at the `You:` prompt. Type `quit` or `exit` to stop.

## Train the model

```powershell
python main.py train
```

The default training command combines:

- `data/instructions.json`: prompt paraphrases and verified responses across
  conversation, programming, science, writing, productivity, security, and
  model concepts;
- `data/training.txt`: general prose for the creative generator.

It writes the trained retrieval weights, instruction responses, and n-gram
counts to `model.json`. To teach a new topic, add an object like this to the
instruction file and retrain:

```json
{
  "topic": "example_topic",
  "prompts": [
    "first way a user might ask",
    "a different phrasing of the same question"
  ],
  "responses": [
    "A concise, accurate response."
  ]
}
```

Use several realistic paraphrases per topic. Evaluate with separate questions
that are not exact copies of the training prompts.

Custom paths and generator order are supported:

```powershell
python main.py train `
  --instructions data/instructions.json `
  --data data/training.txt `
  --model model.json `
  --order 5
```

## Creative text continuation

The n-gram component remains available for probabilistic continuation:

```powershell
python main.py generate `
  --prompt "Language models " `
  --length 300 `
  --temperature 0.8 `
  --seed 42
```

Lower temperature is more predictable; higher temperature adds variety. A
fixed seed makes output reproducible.

## Inspect and test

```powershell
python main.py info
python -m unittest -v
```

The checked-in model contains 63 response topics, 294 instruction examples,
and 56,815 characters of combined generator training text.

## How query answering works

Training converts every example prompt into TF-IDF word and phrase features.
At query time, cosine similarity selects the closest example. The associated
topic supplies a verified response only when the score meets the confidence
threshold. Basic arithmetic is evaluated by a restricted local calculator.

This produces much more useful query behavior than raw character prediction,
but it is not equivalent to a modern pretrained LLM. It cannot synthesize broad
knowledge or perform deep general reasoning. Reaching that level requires a
large neural transformer, extensive training data and compute, or integration
with an existing pretrained model.
