# Intent-to-Policy

Supplementary code and data for the paper:
**"Intent-to-Policy: An Ontology-Grounded Agentic AI System for Reliable ODRL Generation and Validation"**

> Anonymized for double-blind review.

## Repository Layout

```text
semantic-policy-generation/
├── agents/
│   ├── reasoner/        # conflict detection agent
│   ├── generator/       # ODRL Turtle generation agent
│   └── validator/       # SHACL validation + repair agent
├── data/
│   ├── approved_policies/approved_policies_dataset.json
│   ├── rejected_policies/rejected_policies_dataset.json
│   └── text2policy/text2ttl_GT.jsonl
├── evaluation/
│   ├── evaluate_reasoning_agent.py
│   ├── evaluate_pipeline.py
│   ├── evaluate_text2ttl_pipeline.py
│   ├── openai-apis/
│   │   ├── example_models.json
│   │   └── custom_models.json   # local, git-ignored
│   └── results/                 # evaluation outputs land here
└── README.md
```

## Requirements

- Linux/macOS
- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) for environment management
- Access to an Azure OpenAI deployment or an OpenAI-compatible API endpoint

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repository root
uv sync
```

## Model Configuration

Evaluation scripts load model endpoints from `evaluation/openai-apis/custom_models.json`.

```bash
cp evaluation/openai-apis/example_models.json evaluation/openai-apis/custom_models.json
```

Edit each entry (`base_url`, `model_id`, `api_key`) with your credentials. If `--model-id` is omitted, the first entry is used; otherwise it must match an existing `model_id`.

## Reproducing Paper Experiments

All commands use deterministic settings (`temperature=0.0`). Results are written to `evaluation/results/` as JSON.

### 1. Reasoning agent (139 policies)

```bash
uv run python evaluation/evaluate_reasoning_agent.py
uv run python evaluation/evaluate_reasoning_agent.py --model-id deepseek-chat
```

### 2. End-to-end pipeline on approved policies

```bash
uv run python evaluation/evaluate_pipeline.py --dataset-size -1
uv run python evaluation/evaluate_pipeline.py --model-id gpt-oss-120b --dataset-size -1
```

### 3. Input-to-policy pipeline (`text2ttl_GT.jsonl`)

```bash
uv run python evaluation/evaluate_text2ttl_pipeline.py --dataset-size -1
uv run python evaluation/evaluate_text2ttl_pipeline.py --dataset-size -1 --respect-reasoner-gate
```

## Output Files

Written to `evaluation/results/`:

- `agent_results.json` — reasoning agent outputs
- `*_pipeline_metrics.json`, `*_pipeline_results.json` — end-to-end pipeline
- `*_text2ttl_pipeline_metrics.json`, `*_text2ttl_pipeline_details.json` — input-to-policy pipeline

## Datasets

**Benchmark A — Reasoning + Pipeline (139 policies)**
- `data/approved_policies/approved_policies_dataset.json` (72)
- `data/rejected_policies/rejected_policies_dataset.json` (67)

Rejected-split conflict types: vagueness (17), temporal (21), spatial (3), action hierarchy (13), role hierarchy (7), circular dependency (6).

**Benchmark B — Input-to-Policy (50 samples)**
- `data/text2policy/text2ttl_GT.jsonl`

## License

MIT.
