# agents/reasoner/reasoner_agent.py (Works with both Azure and local)

"""
ODRL Reasoner Agent - Single LLM Call Version
Works with Azure OpenAI or OpenAI-compatible endpoints (Fraunhofer, Ollama, etc.)
"""

from typing import Optional, Literal
from datetime import datetime, timezone
import json
import re
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pathlib import Path
import sys
import logging

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agents.reasoner.conflict_types import ConflictType

logger = logging.getLogger(__name__)


# ===== MODEL DEFINITIONS =====

class DetectedIssue(BaseModel):
    """Individual detected issue"""
    category: ConflictType
    conflict_type: Optional[ConflictType] = None
    severity: Literal["high", "low"]
    field: str
    policy_id: Optional[str] = None
    message: str
    suggestion: Optional[str] = None
    detected_in_phase: int


class ReasoningResult(BaseModel):
    """Complete reasoning output"""
    decision: Literal["approve", "reject"]
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[DetectedIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    reasoning: str
    risk_level: Literal["critical", "high", "medium", "low"]


# ===== PROMPT =====
SINGLE_SHOT_REASONING_PROMPT = """
You are a semantics-grounded ODRL Policy Reasoner.

Your task is to detect semantic conflicts and policy-quality issues BEFORE ODRL generation.
Do NOT rely on intuition or generic language associations when a formal procedure is available.
Use explicit symbolic procedures whenever possible.

CURRENT DATE: {current_date}

If a hierarchy, taxonomy, or scope relation is NOT explicitly given in the policy text or the semantic facts above,
do NOT invent custom facts. You may use standard ODRL action inclusions explicitly listed in this prompt.
If evidence is insufficient, report uncertainty instead of hallucinating relations.

==================================================
I. CORE DISTINCTION
==================================================

You must distinguish between:

1. semantic_conflict
   = a formal incompatibility, contradiction, or unsatisfiable conjunction
   between rules or constraints.

2. quality_issue
   = a policy defect that reduces enforceability or precision
   but is not itself a formal deontic contradiction
   (e.g. vagueness, unenforceable mental-state requirements, missing denial path).

A policy may be rejected because of semantic_conflict OR because of critical quality issues.
But do not label every quality issue as a formal conflict.

==================================================
II. NORMALIZED RULE MODEL
==================================================

Normalize the policy set into rules.

Each normalized rule r must be represented conceptually as:
- rule_id
- policy_id
- deontic_type ∈ {{permission, prohibition, duty}}
- assignee_scope = asge(r)
- assigner_scope = asgr(r) if present
- action = act(r)
- target_scope = tgt(r)
- constraints = φ(r), partitioned by domain:
    - numeric
    - temporal
    - spatial
    - concept
    - other

If one policy contains multiple permissions/prohibitions/duties, treat them as separate normalized rules.

==================================================
III. FORMAL RULE CONFLICT PREDICATE
==================================================

A rule-level deontic conflict conflict(r1, r2) holds iff ALL FIVE conditions hold:

(i)   r1 is a permission and r2 is a prohibition, or vice versa
(ii)  act(r1) and act(r2) are comparable in the action hierarchy:
      act(r1) ⊑ act(r2) OR act(r2) ⊑ act(r1)
(iii) asge(r1) ∩ asge(r2) ≠ ∅
(iv)  tgt(r1) ∩ tgt(r2) ≠ ∅
(v)   φ(r1) ∧ φ(r2) is satisfiable in at least one situation σ

If ANY of (i)-(v) fails, do NOT report a rule-level deontic conflict.

This is a STATIC CHECK.
No runtime world-state simulation is required.

==================================================
IV. FORMAL PROCEDURES BY DOMAIN
==================================================

------------------------------
A. ACTION HIERARCHY
------------------------------
Actions form a subsumption hierarchy via odrl:includedIn.

Procedure:
1. Compute the reflexive transitive closure of odrl:includedIn.
2. Check whether a1 ⊑ a2 or a2 ⊑ a1.
3. Use direction carefully:
   - a permission on a broader action may extend to included narrower actions only if the hierarchy semantics supports it.
   - a prohibition on a broader action propagates downward to included narrower actions.
4. Standard example facts that may be used if relevant:
   odrl:print ⊑ odrl:reproduce ⊑ odrl:use

Do not assume undocumented custom action hierarchies.

------------------------------
B. PARTY AND ASSET SCOPE
------------------------------
Procedure:
- conflict at party level iff asge(r1) ∩ asge(r2) ≠ ∅
- conflict at asset level iff tgt(r1) ∩ tgt(r2) ≠ ∅

Use:
- explicit identity match
- explicit subset/superset facts
- explicit hierarchy facts from semantic context
- bounded universal sets only if the scope is clearly defined

Examples:
- "all registered users of Project X" is bounded and may be measurable
- "everyone" without organizational or contractual boundary is unbounded and vague

------------------------------
C. NUMERIC CONSTRAINTS
------------------------------
Domains include:
- count
- payAmount
- percentage
- resolution
- other scalar constraints

Satisfaction:
- count, resolution ∈ N
- payAmount, percentage ∈ Q≥0 unless explicitly defined otherwise

Conflict procedure:
- Check conjunction satisfiability in Linear Integer Arithmetic / linear scalar inequalities
- Unsatisfiable iff no scalar value satisfies all constraints simultaneously

Examples:
- count ≤ 5 AND count ≥ 10  -> unsatisfiable
- percentage = 100 AND percentage ≤ 50 -> unsatisfiable
- payAmount < 0 -> invalid domain usage unless explicitly allowed

Important:
- Numeric unsatisfiability within one rule = intra-rule semantic conflict
- Numeric overlap across permission/prohibition rules contributes to condition (v) of rule conflict

------------------------------
D. TEMPORAL CONSTRAINTS
------------------------------
Domains include:
- dateTime
- timeInterval
- elapsedTime

Use TWO procedures:

1. Qualitative temporal reasoning:
   Apply Allen's Interval Algebra to interval pairs.

Relevant relations:
- no conflict by overlap test alone: before, after
- boundary-touch only: meets, met-by
- potentially conflicting overlap relations:
  equals, overlaps, starts, during, finishes,
  overlapped-by, started-by, contains, finished-by

For contradictory permission/prohibition rules:
- temporal overlap is relevant only if the intervals stand in one of the overlap/containment relations above.

2. Quantitative temporal satisfiability:
   Apply arithmetic reasoning to endpoints and durations.

Unsatisfiable examples:
- dateTime ≤ 2024-01-01 AND dateTime ≥ 2025-01-01
- available after 2025 AND usable only in 2024
- delete before 2024-07-01 AND retain until 2024-12-31
- elapsedTime ≤ 7 days AND elapsedTime ≥ 30 days

Expired policy:
- if the policy validity end date is strictly earlier than {current_date}, report temporal_expired_policy

Important:
- Allen's algebra determines topological relation only
- Numeric satisfiability still requires endpoint/duration reasoning

------------------------------
E. SPATIAL CONSTRAINTS
------------------------------
Domains include:
- spatial
- spatialCoordinates

Use TWO procedures:

1. Topological reasoning:
   Apply RCC-8 style reasoning over regions.

For simultaneous applicability, regions may be compatible if they overlap or one contains the other:
- EQ, PO, TPP, NTPP, and converses

If regions are:
- DC (disconnected), or
- only externally connected without usable overlap,
then the conjunction may be spatially unsatisfiable.

2. Hierarchy reasoning:
   Apply geographic subsumption over explicit GeoNames-like containment facts.

Examples:
- Berlin ⊑ Germany
- Germany ⊑ EU

Interpretation:
- Permission in Germany + prohibition in EU can create a deontic conflict if all other scopes overlap,
  because Germany is contained in EU and both rules can apply in Germany.
- If two spatial scopes are disjoint, they do not create a cross-rule deontic conflict,
  though they may reveal an impossible conjunction inside a single rule.

------------------------------
F. CONCEPT CONSTRAINTS
------------------------------
Domains include:
- purpose
- industry
- language
- deliveryChannel
- other taxonomy-governed concept scopes

Satisfaction:
- sem(l) ⊑ v in the relevant taxonomy

Conflict procedure:
- Apply EL-style subsumption / taxonomy overlap reasoning
- The conjunction is unsatisfiable iff the concept scopes are disjoint,
  i.e. there is no common subclass or overlap compatible with both constraints

Examples:
- purpose = research AND purpose = nonResearchOnly -> unsatisfiable if disjoint
- language = GermanOnly AND language = EnglishOnly -> unsatisfiable unless multilingual interpretation is explicitly allowed

Do not invent taxonomy disjointness unless it is explicit or standard in the provided facts.

==================================================
V. SIX ANALYSIS PHASES
==================================================

PHASE 1 — NORMALIZATION
Extract and normalize all rules from the policy set.
For each rule, identify:
- deontic type
- action
- assignee scope
- target scope
- constraints by domain

PHASE 2 — VAGUENESS / UNBOUNDED SCOPE (q_e = ⊤)
Detect critical quality issues caused by objectively untestable or unbounded expressions.

Flag as unmeasurable_terms ONLY when at least one of the following holds:
- no objective criterion exists for deciding whether the condition is satisfied
- actor scope is unbounded and cannot be operationalized
- asset scope is universal or unconstrained
- action/condition relies on subjective judgment with no measurable proxy

Examples that SHOULD usually be flagged:
- "urgent", "soon", "responsibly", "when necessary", "if important"
- "everyone can access everything"
- "nobody can do anything"

Examples that should NOT automatically be flagged:
- "all registered users in project Alpha"
- "all employees with active contracts"
- any universal quantifier that is explicitly bounded and operationalizable

PHASE 3 — INTRA-RULE SATISFIABILITY
Within each normalized rule, check whether the conjunction of its own constraints is satisfiable.

Check:
- numeric unsatisfiability
- temporal impossible sequence
- temporal endpoint inconsistency
- spatial disjointness
- concept taxonomy disjointness
- multiple constraints on the same operand that jointly cannot hold

Do NOT rely only on metadata.has_conflicting_constraints.
Use the actual constraints and formal tests.

PHASE 4 — HIERARCHY AND SCOPE CLOSURE
Compute:
- action hierarchy closure
- role / party hierarchy closure if explicit facts exist
- asset scope overlap / containment
- party scope overlap / containment

Check:
- action_hierarchy_conflict
- action_subsumption_conflict
- role_hierarchy_conflict
- party_specification_inconsistency

PHASE 5 — INTER-RULE DEONTIC CONFLICT
For every pair of normalized rules, apply the five-condition conflict predicate.

Only report a semantic deontic conflict if all five conditions hold.

When reporting, specify:
- which conditions (i)-(v) were satisfied
- what hierarchy/subsumption fact was used
- what scope intersection was used
- which domain reasoning established satisfiable overlap

PHASE 6 — DUTY / WORKFLOW DEPENDENCIES
Detect:
- circular_approval_dependency
- workflow_cycle_conflict
- incomplete workflow handling
- impossible duty ordering

Examples:
- A requires B, B requires C, C requires A
- "approved requests proceed" with no denial handling -> quality issue, not necessarily semantic conflict

==================================================
VI. DECISION AND RESOLUTION
==================================================

Decision logic:
- REJECT if any semantic_conflict is found
- REJECT if any critical quality_issue is found
- APPROVE only if no semantic conflicts exist and no critical quality issue blocks enforceability

Conflict resolution:
If conflict(r1, r2) holds, also report the recommended resolution outcome:
- if conflict_resolution_mode = odrl:perm      -> permission prevails
- if conflict_resolution_mode = odrl:prohibit  -> prohibition prevails
- if conflict_resolution_mode = odrl:invalid   -> policy becomes void
- if conflict_resolution_mode is absent        -> default to odrl:invalid

==================================================
VII. RISK LEVELS
==================================================

- Critical: semantic contradiction with clear formal evidence; unbounded vagueness blocking operationalization
- High: strong conflict with partial but sufficient hierarchy/scope evidence
- Medium: likely issue, but some hierarchy/taxonomy fact is missing
- Low: minor quality issue only

==================================================
VIII. OUTPUT FORMAT
==================================================

Return valid JSON only.

{{
  "decision": "APPROVE" | "REJECT",
  "confidence": 0.0,
  "risk_level": "Critical" | "High" | "Medium" | "Low",
  "policies_analyzed": <integer>,
  "normalized_rules_count": <integer>,
  "issues": [
    {{
      "issue_kind": "semantic_conflict" | "quality_issue",
      "category": "<short category>",
      "conflict_type": "<enum or None>",
      "severity": "Critical" | "High" | "Medium" | "Low",
      "detected_in_phase": 1 | 2 | 3 | 4 | 5 | 6,
      "policy_id": "<policy id or array>",
      "rule_id": "<rule id or array>",
      "field": "<field or constraint domain>",
      "formal_test": "<Allen | LIA | RCC-8 | EL | set-intersection | hierarchy-closure | workflow-cycle | None>",
      "evidence": {{
        "deontic_pair": "<permission/prohibition/duty>",
        "action_relation": "<a1 ⊑ a2 | a2 ⊑ a1 | equal | none>",
        "party_overlap": "<non-empty | empty | uncertain>",
        "asset_overlap": "<non-empty | empty | uncertain>",
        "constraint_satisfiability": "<satisfiable | unsatisfiable | uncertain>",
        "resolution_outcome": "<odrl:perm | odrl:prohibit | odrl:invalid | None>"
      }},
      "message": "<clear explanation>",
      "suggestion": "<specific repair suggestion>"
    }}
  ],
  "recommendations": [
    "<repair or rewrite recommendation>"
  ],
  "reasoning_summary": [
    "<brief summary of the key formal findings, not chain-of-thought>"
  ]
}}

==================================================
IX. CONFLICT TYPE ENUM
==================================================

Use the MOST SPECIFIC matching type.

Allowed values:
1. unmeasurable_terms
2. numeric_constraint_unsat
3. temporal_overlap
4. temporal_expired_policy
5. temporal_impossible_sequence
6. spatial_hierarchy_conflict
7. spatial_overlap_conflict
8. concept_taxonomy_conflict
9. action_hierarchy_conflict
10. action_subsumption_conflict
11. role_hierarchy_conflict
12. party_specification_inconsistency
13. circular_approval_dependency
14. workflow_cycle_conflict
15. None

Important:
- Use numeric_constraint_unsat for scalar inequality contradictions
- Use concept_taxonomy_conflict for purpose/industry/language/deliveryChannel disjointness
- Use temporal_overlap only for contradictory permission/prohibition rules with temporally overlapping applicability
- Use spatial_hierarchy_conflict when one geographic scope subsumes the other and makes contradictory rules jointly applicable
- Use spatial_overlap_conflict for non-hierarchical geographic overlap leading to contradictory applicability

==================================================
X. POLICY TEXT TO ANALYZE
==================================================
{policy_text}

CRITICAL INSTRUCTIONS:
1. Normalize first, then reason.
2. Do not report a rule-level deontic conflict unless all five conflict conditions hold.
3. Separate semantic_conflict from quality_issue.
4. Use domain-specific formal tests:
   - numeric -> LIA / scalar inequality satisfiability
   - temporal -> Allen + endpoint/duration arithmetic
   - spatial -> RCC-8 + geographic subsumption
   - concept -> EL-style taxonomy overlap/disjointness
5. Do not over-flag bounded universal quantifiers as vague.
6. Do not hallucinate hierarchies or disjointness not supported by the input or explicit semantic facts.
7. Return valid JSON only.
"""


# ===== SINGLE-SHOT REASONER =====

class Reasoner:
    """
    Single-shot ODRL Policy Contradiction Detector
    Works with Azure OpenAI or OpenAI-compatible endpoints
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-2024-11-20",
        temperature: float = 0.0,
        # Azure-specific (optional)
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        # OpenAI-compatible (optional)
        base_url: Optional[str] = None
    ):
        """
        Initialize Reasoner with either Azure or OpenAI-compatible endpoint
        
        Azure usage:
            Reasoner(
                api_key="...",
                api_version="2024-10-01-preview",
                azure_endpoint="https://....openai.azure.com/",
                model="gpt-4o-2024-11-20"
            )
        
        Local/Fraunhofer usage:
            Reasoner(
                api_key="dummy",
                base_url="http://dgx.fit.fraunhofer.de/v1",
                model="deepseek-r1:70b"
            )
        """
        
        # Determine which client to use
        if azure_endpoint or api_version:
            # Use Azure OpenAI
            self.llm = AzureChatOpenAI(
                api_key=api_key,
                api_version=api_version or "2024-10-01-preview",
                azure_endpoint=azure_endpoint,
                model=model,
                temperature=temperature
            )
            self.endpoint_type = "Azure"
        else:
            # Use OpenAI-compatible endpoint
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature
            )
            self.endpoint_type = "OpenAI-compatible"
    
    def reason(self, policy_text: str) -> dict:
        """
        Execute single-shot conflict detection on natural language policy text
        
        Args:
            policy_text: Natural language policy description
            
        Returns:
            Complete reasoning result with decision and issues
        """
        
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        logger.info("=" * 60)
        logger.info(f"STARTING CONFLICT DETECTION ({self.endpoint_type})")
        logger.info(f"Date: {current_date}")
        logger.info(f"Input length: {len(policy_text)} characters")
        logger.info("=" * 60)
        
        # Format prompt
        prompt = SINGLE_SHOT_REASONING_PROMPT.format(
            current_date=current_date,
            policy_text=policy_text
        )
        
        # Single LLM call
        logger.info("[LLM] Invoking single comprehensive analysis...")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse response
        result = self._parse_response(response.content)
        
        logger.info("=" * 60)
        logger.info("REASONING COMPLETE")
        logger.info(f"Decision: {result['decision'].upper()}")
        logger.info(f"Confidence: {self._format_confidence_percent(result.get('confidence'))}")
        logger.info(f"Risk Level: {result['risk_level'].upper()}")
        logger.info(f"Total Issues: {len(result['issues'])}")
        logger.info("=" * 60)
        
        return result

    @staticmethod
    def _normalize_conflict_type(raw_value: Optional[str]) -> Optional[ConflictType]:
        """Normalize common conflict label aliases emitted by different models."""
        if not raw_value:
            return None

        normalized = str(raw_value).strip().lower()
        aliases = {
            "temporal_overlap": ConflictType.TEMPORAL_OVERLAP.value,
            "temporal_conflict": ConflictType.TEMPORAL_OVERLAP.value,
            "expired_policy": ConflictType.TEMPORAL_EXPIRED.value,
            "temporal_expired": ConflictType.TEMPORAL_EXPIRED.value,
            "spatial_conflict": ConflictType.SPATIAL_HIERARCHY.value,
            "action_conflict_generic": ConflictType.ACTION_CONFLICT.value,
            "actor_conflict": ConflictType.PARTY_INCONSISTENCY.value,
            "cross_policy_conflict": ConflictType.PARTY_INCONSISTENCY.value,
            "usage_limit_conflict": ConflictType.TEMPORAL_OVERLAP.value,
            "role_conflict": ConflictType.ROLE_HIERARCHY.value,
            "circular_dependency": ConflictType.CIRCULAR_DEPENDENCY.value,
            "vague_term": ConflictType.UNMEASURABLE.value,
            "vague_terms": ConflictType.UNMEASURABLE.value,
            "ambiguous": ConflictType.PARTY_INCONSISTENCY.value,
            "conflict": ConflictType.ACTION_CONFLICT.value,
            "constraint_conflict": ConflictType.ACTION_CONFLICT.value,
            "overly_broad_policy": ConflictType.VAGUE_BROAD.value,
            "overly_broad": ConflictType.VAGUE_BROAD.value,
            "unenforceable": ConflictType.UNMEASURABLE.value,
            "technical_impossibility": ConflictType.ACTION_CONFLICT.value,
            "circular_approval_dependency": ConflictType.CIRCULAR_DEPENDENCY.value,
            "workflow_cycle": ConflictType.WORKFLOW_CYCLE.value,
        }
        normalized = aliases.get(normalized, normalized)

        try:
            return ConflictType(normalized)
        except ValueError:
            return None

    @staticmethod
    def _coerce_confidence(value: object, default: float = 0.5) -> float:
        """Convert model confidence to a safe float in [0.0, 1.0]."""
        if value is None:
            return default
        try:
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return default
                # Accept "85%" and convert to 0.85.
                if text.endswith("%"):
                    raw = float(text[:-1].strip())
                    numeric = raw / 100.0
                else:
                    numeric = float(text)
            else:
                numeric = float(value)

            # If model returns 85 instead of 0.85, treat as percentage.
            if numeric > 1.0:
                if numeric <= 100.0:
                    numeric = numeric / 100.0
                else:
                    return 1.0

            if numeric < 0.0:
                return 0.0
            if numeric > 1.0:
                return 1.0
            return numeric
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_policy_id(value: object) -> Optional[str]:
        """Normalize policy_id: lists/tuples use the first element as str (per prompt schema)."""
        if value is None or value == "":
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            first = value[0]
            return None if first is None else str(first)
        return str(value)

    @staticmethod
    def _coerce_detected_phase(value: object) -> int:
        """LLMs often emit floats; JSON may use null."""
        if value is None:
            return 0
        try:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            text = str(value).strip()
            if not text:
                return 0
            return int(float(text))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _format_confidence_percent(cls, value: object) -> str:
        """Format confidence robustly for logs, never raising."""
        return f"{cls._coerce_confidence(value):.0%}"
    
    def _parse_response(self, content: str) -> dict:
        """Parse LLM response into structured result"""
        
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object in content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
        
        try:
            result = json.loads(json_str)
            
            # Convert issues to DetectedIssue objects
            issues = []
            for issue_dict in result.get("issues", []):
                try:
                    # Prompt asks for both category and conflict_type; models often fill only one.
                    raw_signal = None
                    for key in ("conflict_type", "category"):
                        candidate = issue_dict.get(key)
                        if candidate in (None, "", "None", "none", "null", "NULL"):
                            continue
                        raw_signal = candidate
                        break
                    if raw_signal is None:
                        continue

                    resolved_conflict = self._normalize_conflict_type(raw_signal)
                    if not resolved_conflict:
                        logger.error(
                            "Unknown conflict_type in issue: "
                            f"signal={raw_signal!r}, issue={issue_dict}"
                        )
                        continue

                    severity = str(issue_dict.get("severity", "low")).lower().strip()
                    if severity in ("critical", "high"):
                        severity = "high"
                    else:
                        severity = "low"

                    field_val = issue_dict.get("field") or "unknown"
                    message_val = issue_dict.get("message") or "Conflict detected"
                    suggestion_val = issue_dict.get("suggestion")
                    if suggestion_val is not None:
                        suggestion_val = str(suggestion_val)

                    issues.append(DetectedIssue(
                        category=resolved_conflict,
                        conflict_type=resolved_conflict,
                        severity=severity,
                        field=str(field_val),
                        policy_id=self._coerce_policy_id(issue_dict.get("policy_id")),
                        message=str(message_val),
                        suggestion=suggestion_val,
                        detected_in_phase=self._coerce_detected_phase(
                            issue_dict.get("detected_in_phase", 0)
                        ),
                    ))
                except Exception as e:
                    logger.error(f"Failed to parse issue: {e}")
                    continue
            
            decision = str(result.get("decision", "reject")).lower()
            if decision not in {"approve", "reject"}:
                decision = "reject"

            risk_level = str(result.get("risk_level", "medium")).lower()
            if risk_level not in {"critical", "high", "medium", "low"}:
                risk_level = "medium"

            return {
                "decision": decision,
                "confidence": self._coerce_confidence(result.get("confidence", 0.5)),
                "risk_level": risk_level,
                "reasoning": result.get("reasoning", ""),
                # mode="json" ensures enums are serialized as their string values.
                "issues": [issue.model_dump(mode="json") for issue in issues],
                "recommendations": result.get("recommendations", [])
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Content: {content[:1000]}...")
            
            # Fallback response
            return {
                "decision": "reject",
                "confidence": 0.5,
                "risk_level": "high",
                "reasoning": f"Failed to parse LLM response. Error: {str(e)}",
                "issues": [],
                "recommendations": ["Manual review required - LLM response parsing failed"]
            }