# Artifacts and versioning

## Committed artifact inventory

| Path | Approximate size | Purpose |
| --- | ---: | --- |
| `artifacts/tokenizer.json` | 0.55 MiB | byte-level BPE vocabulary and tokenizer pipeline |
| `artifacts/smalllm_checkpoint.pt` | 26.1 MiB | transformer config, weights, and training metadata |
| `artifacts/retrieval_index.joblib` | 33.2 MiB | TF-IDF models, sparse matrices, corpus identity |
| `artifacts/training_report.json` | 1.5 KiB | human-readable training metrics and config |
| `artifacts/evaluation_report.json` | 10.5 KiB | latest qualitative evaluation output |

Related committed inputs are approximately 13.5 MiB for
`instructions_large.jsonl` and 20.8 MiB for `knowledge_large.jsonl`.

These binaries are committed directly rather than through Git LFS. Review
repository growth when replacing them; old versions remain in Git history.

## Checkpoint format version 1

`save_checkpoint()` writes a PyTorch payload:

```text
format_version: 1
config: TransformerConfig fields
model_state: state_dict tensors
training_state:
  step, epoch, training_examples, validation_examples,
  validation_loss, parameter_count, corpus_records, seed, corpus_sha256
```

Loading requires version `1` and exact tokenizer/config vocabulary agreement.
`weights_only=True` is used for safer loading.

If the payload schema changes incompatibly:

1. increment `format_version`;
2. update loader validation and provide an intentional migration or clear
   incompatibility error;
3. update tests and this document;
4. regenerate the committed artifact.

## Retrieval format version 3

The joblib payload contains:

```text
version: 3
prompt_vectorizer
response_vectorizer
prompt_matrix
response_matrix
record_count
corpus_sha256
```

The vectorizers and sparse matrices are coupled to scikit-learn/joblib versions.
After dependency upgrades, load the existing index and run retrieval tests. If
serialization compatibility is uncertain, rebuild it and review results.

An incompatible index schema requires a version bump, loader update, rebuild,
tests, and documentation.

## Tokenizer compatibility

Token IDs define checkpoint embedding rows. Never replace `tokenizer.json`
without retraining a compatible transformer checkpoint. Matching vocabulary
size alone is not sufficient if token-to-ID assignments changed.

The required special tokens must exist. Changing their text or semantics also
requires prompt/training/inference changes and a full retrain.

## Training report contract

`training_report.json` is used by root documentation, `/info`, and tests. It
contains:

- proof flags for from-scratch training and absence of pretrained weights;
- parameter/corpus/train/validation counts;
- total and current-run steps;
- resume flag and canonical corpus hash;
- best validation loss and derived perplexity;
- elapsed seconds;
- full model config;
- per-epoch validation history.

Keep it synchronized with the checkpoint selected as best. Do not hand-edit
metrics to make documentation pass.

## Corpus identity

`corpus_sha256()` normalizes all line endings to LF before SHA-256. The
transformer checkpoint/report identity uses the transformer corpus; the
retrieval index identity uses the retrieval corpus.

When a corpus changes, old dependent artifacts should fail rather than load.
Rebuild them in the correct order.

## Safe artifact update checklist

1. Preserve old artifacts until new ones pass all checks.
2. Record dataset source revisions, filters, licenses, seed, and training args.
3. Rebuild corpus-dependent artifacts.
4. Inspect payload versions and hashes.
5. Run tokenizer, checkpoint, retrieval, routing, and count tests.
6. Run qualitative evaluation.
7. Update exact counts and metrics across README and documentation.
8. Confirm no raw downloads, temporary checkpoints, or `.venv` files are staged.
9. Confirm total working size is below 2 GiB and review Git history growth.
10. Stage each intended artifact explicitly.

## Recovery principle

Do not weaken integrity validation to recover from a mismatch. Identify which
source changed, restore the matching source/artifact pair or deliberately
regenerate downstream artifacts, then document the new baseline.
