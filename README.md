# Intent-to-Policy: Ontology-Grounded Agentic AI for ODRL Generation

Anonymous repository for the paper:  
**"Intent-to-Policy: An Ontology-Grounded Agentic AI System for Reliable ODRL Generation and Validation"**

---

## What This Does

Transforms natural language policy requirements into validated [ODRL 2.2](https://www.w3.org/TR/odrl-model/) Turtle via a three-stage multi-agent pipeline:

```
Natural Language → Reasoning Agent → Generation Agent → Validation Agent → Valid ODRL
```

Each agent is specialized and single-shot:

| Agent | Input → Output | Role |
|---|---|---|
| **Reasoning** | `(x, ω) → (x', d, φ)` | Detects 6 conflict types before generation |
| **Generation** | `(x', ω) → y'` | Produces W3C-compliant ODRL 2.2 Turtle |
| **Validation** | `(y', s) → y` | Enforces SHACL shapes; targeted repair (max 3 attempts) |

---

## The Three Problems We Address

1. **Latent semantic conflicts** — spatial hierarchies (Germany ⊂ EU), temporal overlaps, action subsumption (share ⊑ distribute), circular duty chains
2. **LLM vocabulary hallucination** — non-existent ODRL terms like `odrl:readOnly`, mismatched operand-operator pairs
3. **Structural non-compliance** — missing UIDs, invalid constraint triples, SHACL shape violations

---

## Conflict Taxonomy (Reasoning Agent)

The reasoning prompt performs phase-based analysis at three ODRL levels:

| Level | Phase | Conflict Type | Formal Basis |
|---|---|---|---|
| Policy | 1 | Vagueness | `q_e = ⊤` (unbounded scope) |
| Policy | 2 | Role hierarchy | `Manager ⊑ Administrator` |
| Rule | 3 | Action hierarchy | `share ⊑ distribute` |
| Rule | 4 | Circular dependency | `d1 → d2 → d1` |
| Constraint | 5 | Temporal | Allen's 13 interval relations |
| Constraint | 6 | Spatial | Geographic containment axioms |

---

## Repository Structure

```
├── agents/
│   ├── reasoner/
│   │   ├── reasoner_agent.py       # 6-phase conflict detection prompt + Reasoner class
│   │   └── conflict_types.py       # ConflictType enum, detection strategies, examples
│   ├── generator/
│   │   └── generator.py            # Ontology-grounded ODRL generation
│   └── validator/
│       ├── validator_agent.py      # SHACL validation + repair loop
│       └── odrl_validation_tool.py # PySHACL + SPARQL compatibility checks
│
├── data/
│   ├── conflicting_policies/       # 67 conflicting (6 categories)
│   └── valid_policies/             # 83 valid (DRK, IDS, MDS sources)
│
├── evaluation/
│   ├── evaluate_reasoning.py       # Reproduces Table 2 (conflict detection)
│   ├── evaluate_pipeline.py        # Reproduces Table 1 (full pipeline)
│   └── openai-apis/
│       └── example_models.json     # Model config template
│
├── config/                         # SHACL shapes
├── .env.example
└── pyproject.toml
```

---

## Installation

```bash
# Requires Python 3.9+
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/anonymous/ODRL_Policy-Reasoning-Agents.git
cd ODRL_Policy-Reasoning-Agents

uv sync
cp .env.example .env        # Add your API credentials
```

---

## Usage

### Minimal Example

```python
from agents.reasoner.reasoner_agent import Reasoner
from agents.generator.generator import Generator
from agents.validator.validator_agent import ValidatorAgent
from dotenv import load_dotenv
import os

load_dotenv()

creds = dict(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

reasoner  = Reasoner(**creds)
generator = Generator(**creds)
validator = ValidatorAgent(**creds)

policy_text = """
UC4 partners may read and use the medieval manuscript collection
for educational and research purposes until December 31, 2025.
"""

# Stage 1: conflict detection
result = reasoner.reason(policy_text)
if result["decision"] == "reject":
    print("Rejected:", result["issues"])
else:
    # Stage 2: generation
    odrl = generator.generate(policy_text)
    # Stage 3: validation + repair
    final = validator.validate_and_regenerate(
        policy_text=policy_text,
        odrl_turtle=odrl["odrl_turtle"],
        max_attempts=3,
    )
    print("Valid" if final["success"] else "Failed")
    print(final["final_odrl"])
```

### Model Configuration

Copy the template and add your endpoints:

```bash
cp evaluation/openai-apis/example_models.json \
   evaluation/openai-apis/custom_models.json
```

`custom_models.json` is gitignored. The first entry is used by default; override with `--model-id`.

---

## Reproducing Paper Results

```bash
# Table 1 — full pipeline (150 policies, default model)
uv run python evaluation/evaluate_pipeline.py --dataset-size -1

# Table 1 — specific model
uv run python evaluation/evaluate_pipeline.py --model-id gpt-oss-120b

# Table 2 — conflict detection only
uv run python evaluation/evaluate_reasoning_agent.py

# Table 2 — specific model
uv run python evaluation/evaluate_reasoning_agent.py --model-id deepseek-chat
```

Results written to `evaluation/results/{model}_results.json`.

---

## Dataset

150 ODRL policy descriptions with ground-truth labels:

**Conflicting (67):**

| Category | n |
|---|---|
| Vagueness | 17 |
| Temporal | 21 |
| Action Hierarchy | 13 |
| Role Hierarchy | 7 |
| Circular Dependency | 6 |
| Spatial | 3 |
---

**Key finding:** Conflict detection accuracy, not structural compliance, is the primary bottleneck. Structural violations are reliably recovered by SHACL-guided repair; missed conflict decisions are not.

See paper Section 5 for full metrics.

---

## Dependencies

Managed via `uv` / `pyproject.toml`: `langchain`, `langchain-openai`, `pyshacl`, `rdflib`, `pydantic`.

---

## License

MIT

---
