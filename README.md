# Intent-to-Policy: Multi-Agent ODRL Generation System

 **" An Ontology-Grounded Agentic AI System for Reliable ODRL Generation and Validation"**

## Overview

Transforms natural language policy descriptions into W3C ODRL 2.2 compliant policies through three specialized agents:

1. **Reasoning Agent** → Detects semantic conflicts across 6 categories before generation
2. **Generation Agent** → Produces ODRL Turtle with vocabulary grounding
3. **Validation Agent** → Enforces SHACL compliance with targeted regeneration

## Key Results

Evaluation on 150 policies (67 conflicting, 83 valid) across 4 model families:

| Model | Conflict Detection | Generation Success | Final Validation |
|-------|-------------------|-------------------|------------------|
| GPT-4o | High accuracy | All policies succeed | All policies valid |
| GPT-OSS 120B | Good accuracy | Most policies succeed | Most policies valid |
| Llama 3.3 70B | Good accuracy | Most policies succeed | Most policies valid |
| DeepSeek R1 70B | Moderate accuracy | Most policies succeed | Most policies valid |

See paper Section 5 (Evaluation) for detailed metrics.

## Quick Start

### Installation
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <this-repository-url>
cd ODRL_Policy-Reasoning-Agents

# Install dependencies
uv sync
```

### Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API credentials
# For GPT-4o: Configure AZURE_OPENAI_* variables
# For open-weight: Configure LLM_BASE_URL and LLM_MODEL
```

**Minimal Azure OpenAI setup (.env):**
```bash
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-2024-11-20
LLM_MODEL=gpt-4o
```

**Minimal open-weight setup (.env):**
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

# Initialize agents (Azure OpenAI example)
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

# Process natural language policy
policy_text = """
UC4 partners can read and use the medieval manuscript collection 
for educational and research purposes until December 31, 2025.
"""

# Step 1: Detect conflicts
result = reasoner.reason(policy_text)
if result["decision"] == "REJECTED":
    print(f"⚠️ Conflicts detected: {result['feedback']}")
    exit()

# Step 2: Generate ODRL Turtle
odrl = generator.generate(policy_text)

# Step 3: Validate with automatic fixing (max 3 attempts)
final = validator.validate_and_regenerate(
    policy_text=policy_text,
    odrl_turtle=odrl["odrl_turtle"],
    max_attempts=3
)

if final["success"]:
    print("✅ Valid ODRL generated!")
    print(final["final_odrl"])
else:
    print("❌ Validation failed after 3 attempts")
```

## Reproducing Paper Results

### Table 1: Conflict Detection Performance
```bash
# Evaluate Reasoning Agent on all 150 policies

# GPT-4o (Azure)
uv run python evaluation/evaluate_reasoning.py

# Open-weight models (requires compatible endpoint)
uv run python evaluation/evaluate_reasoning.py --model deepseek-r1:70b
uv run python evaluation/evaluate_reasoning.py --model gpt-oss:120b
uv run python evaluation/evaluate_reasoning.py --model llama3.3:70b
```

Results saved to: `evaluation/results/{model}_reasoning_results.json`

### Table 2: Complete Pipeline Performance
```bash
# Evaluate all three agents (Reasoning → Generation → Validation)

# GPT-4o end-to-end
uv run python evaluation/evaluate_pipeline.py

# Multi-model comparison (all 4 models)
uv run python evaluation/evaluate_multi_models.py
```

Results saved to: `evaluation/results/{model}_pipeline_results.json`

### Examining Results
```python
import json

# Load reasoning results
with open("evaluation/results/gpt4o_reasoning_results.json") as f:
    results = json.load(f)

print(f"Overall Accuracy: {results['metrics']['overall_accuracy']}")
print(f"Conflict Detection: {results['metrics']['conflict_detection_score']}")
print(f"Missed Conflicts: {results['metrics']['missed_conflicts']}")
```

## Repository Structure
```
ODRL_Policy-Reasoning-Agents/
├── agents/                      # Multi-agent system implementation
│   ├── reasoner/               # Policy Reasoning Agent
│   │   ├── reasoner_agent.py   # Main agent logic
│   │   └── conflict_types.py   # Conflict type definitions
│   ├── generator/              # ODRL Generation Agent
│   │   └── generator.py        # ODRL Turtle generation
│   └── validator/              # SHACL Validation Agent
│       ├── validator_agent.py  # Validation + regeneration logic
│       └── odrl_validation_tool.py  # PySHACL wrapper
│
├── data/                       # Evaluation dataset
│   ├── conflicting_policies/  # 67 conflicting policies
│   │   ├── vagueness/         # 17 policies
│   │   ├── temporal/          # 21 policies
│   │   ├── spatial/           # 3 policies
│   │   ├── action_hierarchy/  # 13 policies
│   │   ├── role_hierarchy/    # 7 policies
│   │   └── circular_dependency/  # 6 policies
│   └── valid_policies/        # 83 valid policies (DRK, IDS, MDS)
│
├── evaluation/                # Paper reproduction scripts
│   ├── evaluate_reasoning.py  # Table 1 (Conflict Detection)
│   ├── evaluate_pipeline.py   # Table 2 (Pipeline Performance)
│   ├── evaluate_multi_models.py  # Multi-model comparison
│   └── results/               # Saved evaluation results (JSON)
│
├── config/                    # SHACL shapes and LLM configs
├── main.py                    # CLI interface
├── .env.example              # Environment template
├── pyproject.toml            # uv dependencies
└── README.md                 # This file
```

## Dataset Details

**150 ODRL policy descriptions** covering diverse dataspace scenarios:

**Conflicting Policies (67 total):**
- Vagueness: 17 policies (overly broad, unimplementable rules)
- Temporal: 21 policies (overlapping time constraints)
- Spatial: 3 policies (geographic hierarchy conflicts)
- Action Hierarchy: 13 policies (permission/prohibition on related actions)
- Role Hierarchy: 7 policies (inconsistent party specifications)
- Circular Dependency: 6 policies (circular approval chains)

**Valid Policies (83 total):**
- Source: DRK (German Culture Dataspace), IDS (International Data Spaces), Mobility Data Space
- Used for testing Generation and Validation agents

All policies are in JSON format with natural language descriptions.

## Implementation Details

**Architecture Choices:**
- **Single-shot prompting:** One LLM call per agent (cost-efficient)
- **Pydantic validation:** Type-safe structured outputs with JSON Schema
- **Deterministic SHACL:** PySHACL with RDFS inference for structural validation
- **SPARQL semantic queries:** Operator-operand compatibility checks
- **Targeted regeneration:** Fixes only SHACL violations, preserves policy semantics (max 3 attempts)

**Agent Specialization:**
- Reasoning Agent: Systematic 6-phase conflict detection
- Generation Agent: ODRL vocabulary grounding with structural templates
- Validation Agent: SHACL enforcement with learning-based regeneration

See paper Section 4 (Methodology) for architectural details.

## Requirements

- **Python:** 3.9 or higher
- **API Access:** Azure OpenAI (for GPT-4o) OR OpenAI-compatible endpoint (for open-weight models)
- **Dependencies:** Managed by `uv` (see `pyproject.toml`)

**Core Dependencies:**
- `langchain` & `langchain-openai`: LLM integration
- `pyshacl`: SHACL validation with RDFS inference
- `rdflib`: RDF graph manipulation
- `pydantic`: Structured output validation
- `python-dotenv`: Environment configuration

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'agents'`
**Fix:** Run commands with `uv run python` instead of `python`

**Issue:** SHACL validation fails with timeout
**Fix:** Increase `SHACL_TIMEOUT` in `.env` (default: 30 seconds)

**Issue:** API rate limits
**Fix:** Add delays between requests or use different API endpoint

**Issue:** Open-weight models unavailable
**Fix:** Requires access to compatible OpenAI endpoint (e.g., Ollama, vLLM, Fraunhofer server)

## License

MIT License - see LICENSE file

## Contact

For questions about this code repository, please open an issue.

For paper-related questions, please refer to the submission system.

---

**Note:** This is an anonymous repository for peer review. Author information and institutional affiliations are omitted per double-blind review requirements.