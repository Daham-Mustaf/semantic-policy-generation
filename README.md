# Intent-to-Policy

Code repository for the paper:
**"Intent-to-Policy: An Ontology-Grounded Agentic AI System for Reliable ODRL Generation and Validation"**

## What This Repository Provides

This repository implements a three-agent pipeline that converts natural language policy requirements into validated W3C ODRL 2.2 Turtle.

Pipeline:

```text
Natural language policy
  -> Reasoning Agent (conflict detection)
  -> Generation Agent (ODRL Turtle drafting)
  -> Validation Agent (SHACL + repair loop)
  -> Final validated ODRL policy
```

The design addresses three practical barriers in ODRL automation:
- semantic conflicts in natural language requirements,
- vocabulary/ontology hallucinations during generation,
- structural and semantic compliance violations in produced policies.

## Scientific Reproducibility Statement

This README is organized to support artifact-style scientific reproducibility:
- explicit environment and dependency requirements,
- fixed evaluation entrypoints and command lines,
- documented dataset locations and benchmark splits,
- deterministic settings (temperature set to `0.0` in evaluation scripts),
- saved machine-readable outputs in `evaluation/results/`.

What this repository does **not** guarantee:
- bitwise-identical outputs across different provider backends,
- unchanged behavior when model versions or hosted endpoints change.

## Method Overview

Let `x` be a natural language policy input, `omega` semantic context, `s` SHACL shapes:

```text
f_reason   : (x, omega) -> (x', d, phi)
f_generate : (x', omega) -> y'
f_validate : (y', s) -> y
```

Where:
- `d` is a decision in `{approve, reject}` as returned by `Reasoner.reason()` (parsed model output),
- `y'` is draft Turtle,
- `y` is final validated Turtle after up to 3 repair attempts.

## Repository Layout

```text
semantic-policy-generation/
├── agents/
│   ├── reasoner/
│   │   ├── reasoner_agent.py
│   │   └── conflict_types.py
│   ├── generator/
│   │   └── generator.py
│   └── validator/
│       ├── validator_agent.py
│       └── odrl_validation_tool.py
├── data/
│   ├── approved_policies/approved_policies_dataset.json
│   ├── rejected_policies/rejected_policies_dataset.json
│   └── text2policy/text2ttl_GT.jsonl
├── evaluation/
│   ├── evaluate_reasoning_agent.py
│   ├── evaluate_pipeline.py
│   ├── evaluate_text2ttl_pipeline.py
│   ├── model_config_loader.py
│   └── openai-apis/
│       ├── example_models.json
│       └── custom_models.json  # local, git-ignored
└── README.md
```

## Requirements

- Linux/macOS (tested in Linux environments)
- **Python 3.13+** to execute this codebase (`requires-python >= 3.13` in `pyproject.toml`). If your system `python3` is older, use **`uv`**: after `uv sync`, run all documented commands with **`uv run …`** so they use the interpreter and environment uv resolves for this project (typically 3.13), instead of calling `python` directly on the host.
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- Access to either:
  - Azure OpenAI deployment, or
  - an OpenAI-compatible API endpoint

## Setup

```bash
# 1) Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Clone and enter repo
git clone https://github.com/Daham-Mustaf/semantic-policy-generation.git
cd semantic-policy-generation

# 3) Install project dependencies
uv sync
```

## Model Configuration

All evaluation scripts load model endpoints from:
`evaluation/openai-apis/custom_models.json`

Create it from template:

```bash
cp evaluation/openai-apis/example_models.json evaluation/openai-apis/custom_models.json
```

Then edit each model entry (`base_url`, `model_id`, `api_key`) with your actual credentials/endpoints.

Behavior:
- If `--model-id` is omitted, the **first** entry in `custom_models.json` is used. The template’s first entry is `azure-gpt-4.1`; configure it, **reorder** the array so your preferred model is first, or pass `--model-id` explicitly (as in the evaluation examples).
- If `--model-id` is provided, it must match an existing `model_id` entry.

## Quick API Usage

**Previous README snippet issues (now fixed below):**

- `Reasoner.reason()` only yields `decision` values **`approve`** or **`reject`**. Checking `"needs_input"` does nothing because that value is never produced by this agent (evaluation scripts use `needs_input` only as a fallback when a field is missing).
- Example policies with an **end date in the past** relative to the reasoner’s “current date” (see `SINGLE_SHOT_REASONING_PROMPT` in `reasoner_agent.py`) are often classified as **`reject`** (e.g. expired temporal scope), so the sample would raise before generation.
- Entries in `custom_models.json` use **`model_id`**; the Python agents expect the keyword **`model`**. Map the field when building `cfg`.

**How to run:** from the repository root, with dependencies installed (`uv sync`), use the same interpreter uv manages:

```bash
cd semantic-policy-generation
uv sync
uv run python your_script.py
```

If you paste the example into `quick_api_demo.py` at the repo root, `uv run python quick_api_demo.py` is enough.

**Example (inline credentials):**

```python
from agents.reasoner.reasoner_agent import Reasoner
from agents.generator.generator import Generator
from agents.validator.validator_agent import ValidatorAgent

cfg = {
    "api_key": "YOUR_API_KEY",
    # Many OpenAI-compatible servers accept either host root or .../v1; try the form your provider documents.
    "base_url": "https://your-openai-compatible-endpoint/v1",
    "model": "your-model-id",
}

reasoner = Reasoner(**cfg, temperature=0.0)
generator = Generator(**cfg, temperature=0.0)
validator = ValidatorAgent(**cfg, temperature=0.0)

# Use an end date still in the future w.r.t. when you run this, or the reasoner may reject as expired.
policy_text = "UC4 partners may use dataset A for research until 2030-12-31."

reason = reasoner.reason(policy_text)
if reason["decision"] != "approve":
    raise ValueError(f"Rejected by reasoner: {reason['issues']}")

draft = generator.generate(policy_text, policy_id="example_001")
final = validator.validate_and_regenerate(
    policy_text=policy_text,
    odrl_turtle=draft["odrl_turtle"],
    max_attempts=3,
)

print(final["success"])
print(final["final_odrl"])
```

**Example (load `evaluation/openai-apis/custom_models.json`):**

```python
import json
from pathlib import Path

from agents.reasoner.reasoner_agent import Reasoner
from agents.generator.generator import Generator
from agents.validator.validator_agent import ValidatorAgent

repo_root = Path(__file__).resolve().parent  # or Path.cwd() if you always run from repo root
entry = next(
    e
    for e in json.loads((repo_root / "evaluation/openai-apis/custom_models.json").read_text())
    if e["model_id"] == "deepseek-chat"  # or pick the first entry, etc.
)

cfg = {
    "api_key": entry["api_key"],
    "base_url": entry["base_url"].rstrip("/"),
    "model": entry["model_id"],
}

reasoner = Reasoner(**cfg, temperature=0.0)
generator = Generator(**cfg, temperature=0.0)
validator = ValidatorAgent(**cfg, temperature=0.0)

policy_text = "UC4 partners may use dataset A for research until 2030-12-31."
reason = reasoner.reason(policy_text)
if reason["decision"] != "approve":
    raise ValueError(f"Rejected by reasoner: {reason['issues']}")

draft = generator.generate(policy_text, policy_id="example_001")
final = validator.validate_and_regenerate(
    policy_text=policy_text,
    odrl_turtle=draft["odrl_turtle"],
    max_attempts=3,
)
print(final["success"])
print(final["final_odrl"])
```

## Reproducing Core Paper Experiments

### 1) Reasoning Agent Evaluation (139 policies)

```bash
# Evaluate all approved+rejected policies
uv run python evaluation/evaluate_reasoning_agent.py

# Optional: evaluate specific model config entry
uv run python evaluation/evaluate_reasoning_agent.py --model-id deepseek-chat
```

Optional slicing:

```bash
uv run python evaluation/evaluate_reasoning_agent.py --start 0 --limit 50
```

### 2) End-to-End Pipeline on Approved Policies

```bash
# Full approved split
uv run python evaluation/evaluate_pipeline.py --dataset-size -1

# Example: switch model by model_id
uv run python evaluation/evaluate_pipeline.py --model-id gpt-oss-120b --dataset-size -1
```

### 3) Input-to-Policy Pipeline (`text2ttl_GT.jsonl`)

```bash
uv run python evaluation/evaluate_text2ttl_pipeline.py --dataset-size -1

# Optionally enforce reasoner gating
uv run python evaluation/evaluate_text2ttl_pipeline.py --dataset-size -1 --respect-reasoner-gate
```

## Output Artifacts

All evaluation outputs are written to `evaluation/results/` as JSON.

Typical files include:
- reasoning outputs: `agent_results.json`
- pipeline metrics/details: `*_pipeline_metrics.json`, `*_pipeline_results.json`
- text2ttl metrics/details: `*_text2ttl_pipeline_metrics.json`, `*_text2ttl_pipeline_details.json`

These files are the primary artifact interface for downstream analysis and table generation.

## Datasets and Benchmarks

### Benchmark A (Reasoning + Pipeline)
- `data/rejected_policies/rejected_policies_dataset.json`
- `data/approved_policies/approved_policies_dataset.json`
- Total: 139 policies (67 rejected / 72 approved), counted as entries in each file’s `policies` array

Conflict-type distribution in rejected split:
- vagueness: 17
- temporal: 21
- spatial: 3
- action hierarchy: 13
- role hierarchy: 7
- circular dependency: 6

### Benchmark B (Input-to-Policy)
- `data/text2policy/text2ttl_GT.jsonl`
- 50 samples (used for end-to-end structured extraction/pipeline evaluation)

## Alignment with Paper Claims

As reported in the manuscript:
- conflict detection quality is the primary bottleneck,
- generation is generally high-success on reasoner-approved inputs,
- SHACL-guided regeneration improves first-pass validity to high final validity.

Use the commands above to regenerate corresponding stage-level metrics.

## Limitations

- Conflict detection still depends on LLM reasoning quality.
- Spatial reasoning may rely on implicit world knowledge unless explicitly grounded.
- Metrics can vary with endpoint/model updates even under fixed prompts.
- Current benchmark setup focuses on single-policy inputs (cross-policy reasoning is future work).

## Responsible Use

- Do not use generated policies without domain/legal review in production enforcement systems.
- Treat outputs as decision support artifacts, not legal guarantees.
- Keep API keys out of version control (`custom_models.json` is local by design).

## Submission Metadata 

Fill these fields when preparing the final submission package:

- **Authors:** `[Daham M. Mustafa]`, `[Yixin Peng]`, `[Diego Collarana]`, `[Stefan Decker]`
- **Affiliations:** `[RWTH Aachen University / Fraunhofer FIT, Aachen / Sankt Augustin, DE]`
- **Contact Email:** `[daham.mustafa@rwth-aachen.de]`
- **Paper Venue/Track:** `[Conference Name, Track]`
- **Endpoint Freeze Date:** `[10-April-2026]`

## License

MIT License.
