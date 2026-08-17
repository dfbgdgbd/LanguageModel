# Data sources and provenance

SmallLM's tokenizer, retrieval index, and transformer checkpoint are trained
from the data in this repository. No pretrained model weights or pretrained
tokenizer files are used.

## SmallLM curated instructions

`data/instructions.json` contains original prompt-response examples written for
this project. The preparation script expands them into one record per prompt in
`data/instructions_large.jsonl`.

## OpenAssistant Conversations Datasets (OASST1 and OASST2)

- Source: <https://huggingface.co/datasets/OpenAssistant/oasst1>
- Source: <https://huggingface.co/datasets/OpenAssistant/oasst2>
- Project: <https://open-assistant.io/>
- License: Apache License 2.0
- Upstream paper: <https://arxiv.org/abs/2304.07327>

The committed corpus contains a deterministic filtered subset of English
prompt-response pairs. `scripts/prepare_dataset.py` keeps top-ranked assistant
responses from export-ready trees and applies limits for length, quality,
toxicity, sexual content, violence, PII labels, duplicates, emails, credentials,
and private-key material. Every record retains its source, upstream message ID,
and license identifier.

The current artifact contains 6,726 filtered OASST1 pairs, 4,242 filtered
OASST2 pairs, and 294 curated pairs, for 11,262 total instruction-response
scenarios. Identical prompt/response pairs are deduplicated across all sources.

## Databricks Dolly 15k

- Source: <https://huggingface.co/datasets/databricks/databricks-dolly-15k>
- License: Creative Commons Attribution-ShareAlike 3.0 Unported
- License terms: <https://creativecommons.org/licenses/by-sa/3.0/>

Dolly contains instruction/response records written by Databricks employees
across brainstorming, classification, closed and open question answering,
generation, information extraction, summarization, and free-form categories.
After length, sensitive-text, and cross-source duplicate filtering, 13,927 Dolly
records are added to `data/knowledge_large.jsonl` for retrieval. They are not
part of the fixed 11,262-record corpus used for the committed transformer run.
The resulting retrieval corpus contains 25,189 scenarios.

The Dolly-derived records and adaptations remain available under CC BY-SA 3.0;
see [LICENSE-DATA-CC-BY-SA-3.0](LICENSE-DATA-CC-BY-SA-3.0).

The original datasets may contain mistakes or biases despite filtering. Important
outputs should be checked against authoritative sources.
