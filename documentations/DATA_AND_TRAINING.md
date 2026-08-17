# Data and training

## Corpora and roles

| File | Records | Used for |
| --- | ---: | --- |
| `data/instructions.json` | curated source structure | original project-authored prompts/responses |
| `data/instructions_large.jsonl` | 11,262 | tokenizer and transformer training |
| `data/knowledge_large.jsonl` | 25,189 | retrieval index training and inference lookup |

The transformer corpus contains 294 curated pairs, 6,726 filtered OASST1 pairs,
and 4,242 filtered OASST2 pairs. The retrieval corpus adds 13,927 filtered Dolly
records. Dolly is not part of the committed transformer training run.

## JSONL schema

Each prepared record is one UTF-8 JSON object per line:

```json
{
  "id": "source-stable-id",
  "prompt": "User instruction",
  "response": "Assistant response",
  "source": "dataset identifier",
  "topic": "topic or category",
  "license": "license identifier"
}
```

`load_corpus()` requires each record to be an object with nonempty `prompt` and
`response`. Preserve `source`, `id`, and `license` even though inference can
fallback when optional metadata is missing.

## Dataset preparation

`scripts/prepare_dataset.py` performs these steps:

1. expand `data/instructions.json` into individual curated records;
2. load OASST1 and OASST2 train splits;
3. choose English, top-ranked (`rank == 0`), non-deleted assistant messages from
   export-ready trees whose parent is an English prompter message;
4. apply content length, toxicity, sexual content, appropriateness, PII, hate,
   violence, quality, email, credential, and private-key filters;
5. sort upstream results by generated record ID;
6. case-insensitively deduplicate identical prompt/response pairs;
7. write the transformer corpus with LF line endings and compact JSON;
8. load and filter Dolly, combine it with transformer records, deduplicate, and
   write the retrieval corpus.

Important current boundaries:

- prompt length: 4–1,200 characters;
- response length: 20–3,000 characters;
- OASST toxicity: below `0.15`;
- OASST explicit sexual score: below `0.08`;
- not-appropriate, PII, hate, and sexual-content labels: at most `0.25`;
- violence label: at most `0.5`;
- quality label: at least `0.5` when present.

Sensitive-text patterns reject private-key headers, obvious API-key/password
assignments, and email addresses. These filters reduce risk but do not prove a
record is safe, correct, unbiased, or free of personal information.

The Hugging Face dataset revisions are not pinned in code. Output is
deterministic for a fixed upstream revision and library behavior, but a future
download can change. For a formally reproducible new run, pin and document
upstream revisions before preparation.

## Licensing

- Project code and curated data: Apache-2.0 as distributed here.
- OASST1/OASST2: Apache-2.0 with attribution retained in `NOTICE`.
- Dolly-derived retrieval data: CC BY-SA 3.0; preserve
  `LICENSE-DATA-CC-BY-SA-3.0`, provenance, and share-alike obligations.

Read `DATA_SOURCES.md`, `NOTICE`, and both license files before redistributing a
modified corpus. When adding a source, document its exact revision, license,
filtering, record count, and whether it enters transformer training, retrieval,
or both.

## Retrieval training

`train_retrieval_index()` fits word and character TF-IDF components and writes a
compressed joblib artifact. It records the canonical corpus hash and count.
Rebuild the index whenever `knowledge_large.jsonl`, retrieval feature logic, or
artifact version changes.

## Tokenizer training

`train_tokenizer()` creates a new byte-level BPE tokenizer from:

1. the default system prompt;
2. every training prompt;
3. every training response.

It uses minimum frequency `2`, the configured vocabulary size, and all required
special tokens. Rebuilding the tokenizer changes token IDs and invalidates the
old checkpoint. A tokenizer rebuild and transformer-from-scratch run therefore
belong in one artifact update.

## Instruction examples

Training text uses:

```text
<bos><system>
{default system prompt}
<user>
{prompt}
<assistant>
{response}<eos>
```

Sequences are truncated to `max_seq_len`. Examples too short to contain a
response after the prefix are skipped. Labels covering the prompt prefix and
padding are set to `-100`, so cross-entropy trains only on assistant response
tokens.

In the committed run, 11,262 corpus records produced 11,130 usable tokenized
examples after sequence filtering: 10,574 training examples and 556 validation
examples.

## Transformer training loop

- deterministic Python and PyTorch seeds;
- shuffled 95% training / 5% validation split;
- AdamW, default learning rate `3e-4`, weight decay `0.05`;
- linear warmup followed by cosine decay with a `0.1` floor multiplier;
- gradient norm clipping at `1.0`;
- validation over at most 50 batches after each epoch;
- checkpoint saved only when validation loss improves;
- report written after the run with metrics, config, checksum, timing, and
  validation history.

The committed run used 8 epochs and 7,056 steps, reaching validation loss
`4.5536` and perplexity `94.97` in roughly 11,366 seconds on CPU.

## Resume semantics

`--resume` validates checkpoint format, exact architecture config, and corpus
hash before loading weights. It restores model weights and the step/history
metadata used in reporting. It does **not** restore optimizer, scheduler,
dataloader, or random-generator states, so it is not bit-for-bit continuation.

If the resumed run never improves validation loss, the prior best checkpoint
remains. The new report still records the additional run and combined step
count.

## Safe artifact rebuild order

1. install `requirements-data.txt`;
2. document or pin upstream revisions;
3. prepare both corpora;
4. review counts, samples, licenses, and sensitive-content risk;
5. build the retrieval index;
6. rebuild tokenizer if vocabulary/data changes require it;
7. train the transformer;
8. run tests and evaluation;
9. inspect artifact metadata and total size;
10. update all baseline documentation in the same change.
