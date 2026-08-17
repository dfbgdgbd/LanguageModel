# Model and inference

## Transformer architecture

The committed checkpoint uses `TransformerConfig` with:

| Field | Value |
| --- | ---: |
| vocabulary | 8,000 |
| maximum sequence length | 192 |
| model width | 256 |
| transformer layers | 6 |
| attention heads | 8 |
| feed-forward width | 1,024 |
| dropout | 0.1 |
| trainable parameters | 6,836,224 |

`build_model()` uses PyTorch's `TransformerEncoderLayer` with a causal upper
triangular mask, GELU activation, batch-first tensors, and pre-norm
(`norm_first=True`). Token and learned positional embeddings are summed. A final
layer norm feeds a bias-free language-model head whose weights are tied to the
token embedding matrix.

Although `TransformerEncoder` is the PyTorch building block, the causal mask
makes this a decoder-only autoregressive model.

## Tokenizer

`ScratchTokenizer` loads `artifacts/tokenizer.json`, an 8,000-token byte-level
BPE tokenizer trained by this repository. Required control tokens are:

```text
<pad> <unk> <bos> <eos> <system> <user> <assistant>
```

Initialization fails if a required token is absent. Decoding skips special
tokens. The tokenizer vocabulary size must exactly match the checkpoint config.

## Prompt format

The assistant constructs prompts in this order:

```text
<bos><system>
{system prompt}
<user>
{older user message}
<assistant>
{older assistant message}
<user>
{current query}
<assistant>
```

Only the newest `max_history_messages` role/content messages are included. At
generation time, the tokenizer keeps the tail of the prompt so space remains
for output. Reserved output space is the smaller of `max_new_tokens` and half
the configured context length (96 tokens for the committed model).

## Generation algorithm

For every new token:

1. run the model on at most the newest 192 token IDs;
2. mask pad, unknown, BOS, system, user, and assistant control token IDs;
3. apply repetition penalty to unique generated token IDs seen in the newest 64
   generated positions;
4. block candidates that would reproduce a configured generated n-gram;
5. if temperature is `0` or lower, choose `argmax`;
6. otherwise divide logits by temperature, keep the top `k`, softmax, and
   sample using the seeded CPU generator;
7. stop at EOS or `max_new_tokens`;
8. decode only generated IDs.

Changing decoding order changes reproducibility and response quality. Add a
fixed-seed regression test whenever it changes.

## Checkpoint loading

`ScratchTransformerBackend` loads with `torch.load(..., map_location="cpu",
weights_only=True)`. It requires checkpoint `format_version == 1`, constructs
the architecture from the stored config, validates tokenizer vocabulary size,
loads `model_state`, sets evaluation mode, and caps PyTorch threads at eight.

The model is deliberately CPU-only in the current product. Introducing another
device requires explicit device movement for the model, tensors, and generator,
plus packaging and storage review.

## Retrieval model

The retrieval index is trained entirely from local project corpora. Artifact
version 3 stores:

- a `FeatureUnion` prompt vectorizer;
- a prompt sparse matrix;
- a separate response vectorizer and response sparse matrix;
- record count;
- canonical corpus SHA-256.

Prompt features combine:

- word TF-IDF, 1–2 grams, up to 60,000 features, weight `1.0`;
- character-within-word TF-IDF, 3–5 grams, up to 40,000 features, weight `0.45`.

Response text uses word TF-IDF, 1–2 grams, up to 20,000 features. Search adds
prompt and response cosine-like sparse scores, multiplies by a normalized-term
coverage factor (`0.35 + 0.65 * coverage`), and adds `0.2` for an exact domain
term match such as Python, SQL, Git, Windows, Docker, or JSON.

`expand_query()` adds a small fixed synonym set for duplicate, reschedule,
summarize, and compare intents. `normalized_terms()` applies lightweight suffix
normalization; it is not a pretrained stemmer.

## Retrieval integrity

`TrainedRetriever` refuses to load when:

- artifact version is not `3`;
- indexed record count differs from the JSONL corpus;
- canonical corpus SHA-256 differs.

This is intentional. Never bypass these checks to make an old index load. Build
a new index from the new corpus.

## Grounding gate

`generation_is_grounded()` rejects a transformer response when it is too short,
has too little vocabulary diversity, does not overlap enough meaningful query
terms, or repeats too many word trigrams. Common instruction words are excluded
from overlap matching.

The gate is deliberately simple and conservative. It is not a factuality model.
A passing response can still be wrong; a failing response can be useful. Treat
threshold changes as behavior changes and evaluate diverse prompts.

## Deterministic tools

Tools run before retrieval and generation in this order:

1. safe arithmetic using a restricted Python AST;
2. local date/time;
3. word, character, and line counts;
4. JSON validation/pretty printing;
5. C/F, km/mi, and kg/lb conversion;
6. selected professional email templates;
7. fixed guided responses for common programming, finance, HTTP 500, study,
   website-launch, science-report, API-comparison, and naming intents.

The calculator does not execute arbitrary Python. It restricts node types,
operators, exponent size, finite results, and maximum magnitude. Preserve these
safety properties when extending it.

To add a tool, return `ToolResult` only for an unambiguous intent, append the
handler in the correct priority order inside `run_tool()`, and add positive and
near-miss tests.
