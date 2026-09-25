# EvoMem

Type-adaptive retrieval and self-evolving memory architecture for LLM agents — multi-session long-term memory with query profiling, versioned entity-attribute tracking, and error-driven pattern learning. Evaluated on LoCoMo and LongMemEval.

## Repository layout

This repository ships two organized versions of the implementation, each a self-contained `hierarchyv2/` Python package:

| Directory | Contents |
|---|---|
| `final/` | Core implementation (TriMem-AR). |
| `locomo/` | Full implementation **including** the LoCoMo official four-category adaptation (`single-hop`, `open-domain`, `multi-hop`, `temporal`) and the `data/` directory with LoCoMo build scripts and caches. **Use this version to reproduce the LoCoMo results.** |

The two versions share the same core modules; `locomo/` additionally contains the LoCoMo-specific question-type mapping and inference branches (dated 2026-09-19) plus the `data/` assets.

## Installation

```bash
cd final      # or: cd locomo
pip install -r requirements.txt
```

## Configuration

Set the following environment variables before running (no keys are hardcoded in the code):

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | (empty) | API key for the LLM backend (DeepSeek, OpenAI-compatible endpoint). |
| `EMBED_PATH` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model identifier. |
| `DATA_PATH` | `data/locomo10_input_50.json` | Path to the evaluation input JSON. |
| `KM_PATH` | `knowledge_memory.json` | Path for the knowledge-memory store (generated at runtime). |

## Running

From inside `final/` or `locomo/`:

```bash
python -m hierarchyv2.main --data <path_to_input.json> --tag trimem_ar
# optional parallel execution across machines/processes:
python -m hierarchyv2.main --data <input.json> --part 0 --num_parts 4
```

`hierarchyv2.main` is the TriMem-AR entry point; it reads the evaluation JSON, runs the memory-augmented QA pipeline, and prints per-type and overall accuracy.

## Dependencies

`requirements.txt` lists: `numpy`, `openai`, `torch`, `transformers`, `sentence_transformers`, `bert_score`. **Pin exact versions before submission.**

## Notes

- No model weights or API keys are bundled. Provide your own `DEEPSEEK_API_KEY`.
- `knowledge_memory*.json` is produced at runtime and is git-ignored.
- For LoCoMo reproduction, use the `locomo/` directory; `data/build_locomo_official4.py` prepares the LoCoMo inputs.
