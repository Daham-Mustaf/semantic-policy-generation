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