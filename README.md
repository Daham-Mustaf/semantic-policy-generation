# ODRL Policy-Reasoning Agents

Anonymous repository: **"Intent-to-Policy: An Ontology-Grounded Agentic AI System for Reliable ODRL Generation and Validation"**

## Overview

A multi-agent system for transforming natural language policy requirements into validated ODRL (Open Digital Rights Language) policies using specialized LLM agents with ontology-grounded reasoning.

### System Architecture

```
Natural Language → [Reasoning Agent] → [Generation Agent] → [Validation Agent] → Valid ODRL Policy
```

The framework decomposes ODRL policy generation into three specialized agents:

```
f_reason:    (x, ω) → (x', d, φ)    [6-Phase Conflict Detection]
f_generate:  (x', ω) → y'            [Ontology-Grounded ODRL Generation]
f_validate:  (y', s) → y             [SHACL-Based Validation]
```

Where:
- `x`: Natural language policy requirements
- `ω`: Semantic context (ODRL ontology, domain knowledge)
- `x'`: Structured requirements
- `d`: Decision (APPROVED/REJECTED/NEEDS_INPUT)
- `φ`: Conflict feedback
- `y'`: Draft ODRL policy
- `y`: Validated ODRL policy
- `s`: SHACL validation shapes

1. **Semantic conflicts in natural language** (spatial hierarchies, temporal overlaps, action subsumption)
2. **LLM vocabulary hallucination** (generating non-existent ODRL terms)
3. **Structural compliance failures** (violating SHACL constraints)

## Architecture

Three specialized agents working in sequence:

**1. Policy Reasoning Agent**
- Detects six conflict types: vagueness, temporal, spatial, action hierarchy, role hierarchy, circular dependencies
- Rejects contradictory policies before generation
- Structured phase-based analysis with explicit conflict feedback

**2. ODRL Generation Agent**
- Produces W3C ODRL 2.2 compliant Turtle
- Vocabulary grounding through prompt engineering
- Generates mandatory structural declarations and metadata

**3. SHACL Validation Agent**
- Enforces compliance via PySHACL with RDFS inference
- SPARQL-based semantic compatibility checks (operator-operand validation)
- Targeted regeneration preserving policy semantics (max 3 attempts)

## Repository Structure
```
ODRL_Policy-Reasoning-Agents/
├── agents/                      # Multi-agent implementation
│   ├── reasoner/               # Conflict detection agent
│   │   ├── reasoner_agent.py
│   │   └── conflict_types.py
│   ├── generator/              # ODRL generation agent
│   │   └── generator.py
│   └── validator/              # SHACL validation agent
│       ├── validator_agent.py
│       └── odrl_validation_tool.py
│
├── data/                       # Evaluation dataset (150 policies)
│   ├── conflicting_policies/  # 67 conflicting (6 categories)
│   └── valid_policies/        # 83 valid (DRK, IDS, MDS)
│
├── evaluation/                # Paper reproduction scripts
│   ├── evaluate_reasoning.py
│   ├── evaluate_pipeline.py
│   └── results/
│
├── config/                    # SHACL shapes
├── .env.example              # Configuration template
├── pyproject.toml            # Dependencies (uv)
└── README.md
```

## Installation
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/Daham-Mustaf/ODRL_Policy-Reasoning-Agents.git
cd ODRL_Policy-Reasoning-Agents

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API credentials
```

## Quick Start

### Configuration

**Azure OpenAI (GPT-4o):**
```bash
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-2024-11-20
LLM_MODEL=gpt-4o
```

**OpenAI-Compatible Endpoint (Open-Weight Models):**
```bash
LLM_BASE_URL=http://your-endpoint/v1
LLM_API_KEY=your_key
LLM_MODEL=deepseek-r1:70b
```

### Basic Usage
```python
from agents.reasoner.reasoner_agent import Reasoner
from agents.generator.generator import Generator
from agents.validator.validator_agent import ValidatorAgent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize agents
reasoner = Reasoner(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

generator = Generator(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

validator = ValidatorAgent(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Process policy
policy_text = """
UC4 partners can read and use the medieval manuscript collection 
for educational and research purposes until December 31, 2025.
"""

# Step 1: Detect conflicts
result = reasoner.reason(policy_text)
if result["decision"] == "REJECTED":
    print(f"Conflicts: {result['feedback']}")
    exit()

# Step 2: Generate ODRL
odrl = generator.generate(policy_text)

# Step 3: Validate and fix
final = validator.validate_and_regenerate(
    policy_text=policy_text,
    odrl_turtle=odrl["odrl_turtle"],
    max_attempts=3
)

print(" Valid" if final["success"] else " Failed")
print(final["final_odrl"])
```

## Reproducing Paper Results

### Conflict Detection Evaluation (Table 1)
```bash
# GPT-4o
uv run python evaluation/evaluate_reasoning.py

# Open-weight models
uv run python evaluation/evaluate_reasoning.py --model deepseek-r1:70b
uv run python evaluation/evaluate_reasoning.py --model gpt-oss:120b
uv run python evaluation/evaluate_reasoning.py --model llama3.3:70b
```

### Complete Pipeline Evaluation (Table 2)
```bash
# End-to-end evaluation
uv run python evaluation/evaluate_pipeline.py

# Multi-model comparison
uv run python evaluation/evaluate_multi_models.py
```

Results saved to: `evaluation/results/{model}_results.json`

## Dataset

**150 ODRL policy descriptions:**

**Conflicting Policies (67):**
- Vagueness: 17
- Temporal: 21
- Spatial: 3
- Action Hierarchy: 13
- Role Hierarchy: 7
- Circular Dependency: 6

**Valid Policies (83):**
- Sources: DRK (German Culture Dataspace), IDS, Mobility Data Space
- Used for generation and validation testing

## Key Implementation Features

- **Single-shot architecture:** One LLM call per agent
- **Pydantic validation:** Type-safe structured outputs
- **Deterministic SHACL:** PySHACL with RDFS inference
- **SPARQL queries:** Semantic operator-operand compatibility
- **Targeted regeneration:** Fixes syntax while preserving semantics

## Evaluation Summary

Tested on 150 policies across 4 model families (GPT-4o, GPT-OSS 120B, Llama 3.3 70B, DeepSeek R1 70B):

- **Reasoning Agent:** High conflict detection accuracy across all models
- **Generation Agent:** Successful ODRL creation for valid policies
- **Validation Agent:** Effective error recovery through targeted regeneration

See paper Section 5 for detailed metrics and analysis.

## Requirements

- Python 3.9+
- Azure OpenAI API OR OpenAI-compatible endpoint
- Dependencies: langchain, pyshacl, rdflib, pydantic (managed by uv)

## License

MIT License

## Note

This is an anonymous repository for peer review. Author information and institutional affiliations are omitted per double-blind requirements.

For questions, please open an issue in this repository.