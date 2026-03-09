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
    decision: Literal["approve", "reject", "needs_input"]
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[DetectedIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    reasoning: str
    risk_level: Literal["critical", "high", "medium", "low"]


# ===== SINGLE COMPREHENSIVE PROMPT =====

SINGLE_SHOT_REASONING_PROMPT = """
You are a universal ODRL Policy Contradiction Detector.
Your job: Analyze policy sets for ANY logical contradiction, regardless of domain.

## CORE ANALYSIS FRAMEWORK

### CURRENT DATE: {current_date}

### SYSTEMATIC CONTRADICTION DETECTION

You must check for contradictions across these dimensions:

#### 1. INTRA-POLICY CONSTRAINT CONFLICTS

**CRITICAL: Check for mutually exclusive constraints within a SINGLE policy:**

If a policy has multiple constraints with different constraint_group values on the same leftOperand, this indicates a contradiction.

**Rule:** If constraint_group is present and multiple constraints have the same leftOperand but different groups, this is a HIGH severity contradiction.

**Example patterns:**
- Multiple purpose constraints with different constraint_group values
- Multiple count constraints with different constraint_group values  
- Multiple temporal constraints with different constraint_group values

If metadata.has_conflicting_constraints is true, examine the constraints array for conflicts.

#### 2. ACTOR CONTRADICTIONS
**Check if policies conflict on WHO can do something:**

- Same actor given BOTH permission AND prohibition for same action/asset
- Actor restrictions that overlap: "only UC4" vs "all partners" (if UC4 is a partner)
- Role hierarchy conflicts: "managers allowed" + "administrators prohibited" + "all managers are administrators"
- Universal quantifiers: "everyone" vs "nobody", "all users" vs "no users"

**Examples:**
- "UC4 can access dataset" + "UC4 cannot access dataset"
- "Only researchers" + "All registered users" (if researchers are users)
- "Managers allowed" + "All administrators prohibited" + "Managers are administrators"

#### 3. ACTION CONTRADICTIONS
**Check if policies conflict on WHAT can be done:**

- Same action both permitted AND prohibited on same asset
- Action hierarchy conflicts: "can use" vs "cannot read" (if use includes read)
- Contradictory action types: "can modify" + "cannot modify"

**Examples:**
- "Can read document.pdf" + "Cannot read document.pdf"
- "Can use dataset" + "Cannot access dataset" (use requires access)
- "Researchers can modify metadata" + "Metadata must not be modified"

#### 4. ASSET/TARGET CONTRADICTIONS
**Check if policies conflict on WHICH resources:**

- Same asset subject to contradictory rules
- Asset hierarchy conflicts: "all datasets" vs "dataset X prohibited"

**Examples:**
- Permission on dataset:123 + Prohibition on dataset:123

#### 5. TEMPORAL CONTRADICTIONS
**Check if policies conflict on WHEN:**

- Overlapping time windows with contradictory rules
- Expired policies (before current date: {current_date})
- Contradictory time constraints: "until 2025" + "indefinitely"
- Impossible sequences: "available after 2025" + "can use in 2024"

**Examples:**
- "Access 9am-5pm" + "Prohibited 2pm-6pm" (overlap: 2-5pm)
- "Until Jan 1, 2025" + "Indefinitely"
- "Available only after 2025" + "Can use in 2024 for education"
- "Permitted before Jan 1, 2020" (current: {current_date}) → EXPIRED

#### 6. SPATIAL/LOCATION CONTRADICTIONS
**Check if policies conflict on WHERE:**

- Geographic hierarchy conflicts: "allowed in Germany" + "prohibited in EU" (Germany ⊂ EU)
- Overlapping locations with contradictory rules
- Location containment: broader region prohibits what narrower region allows

**Examples:**
- "Access in Germany only" + "Prohibited in all EU countries" (Germany is in EU)
- "Allowed in Berlin" + "Prohibited in Germany" (Berlin ⊂ Germany)

#### 7. QUANTITATIVE/USAGE LIMIT CONTRADICTIONS
**Check if policies conflict on HOW MUCH:**

- Contradictory count limits: "30 times" + "unlimited"
- Contradictory size limits: "max 1024 MiB" + "min 2048 MiB"
- Contradictory bandwidth: "max 20 Mbit/s" + "min 50 Mbit/s"
- Contradictory concurrency: "max 5 connections" + "min 10 connections"
- Percentage conflicts: "aggregate 100%" + "aggregate max 50%"

**Examples:**
- "Use up to 30 times" + "Unlimited access" (for same actor/asset/action)
- "Read max 1024 MiB" + "Must read at least 2048 MiB"
- "Aggregate 100% of File1" + "Aggregate max 50% of File1"

#### 8. PURPOSE/CONSTRAINT CONTRADICTIONS
**Check if policies conflict on WHY/UNDER WHAT CONDITIONS:**

- Contradictory purposes: "for research" + "not for research"
- Overlapping purposes: "educational only" + "any purpose"
- Contradictory constraints: "must inform provider" + "must not inform provider"
- Data handling conflicts: "delete before July 10" + "retain until Dec 31"

**Examples:**
- "For research purpose" + "Prohibited for research"
- "Must inform provider after use" + "Prohibited from informing provider"
- "Delete before 2023-07-10" + "Retain until 2023-12-31"

#### 9. TECHNICAL FEASIBILITY CONTRADICTIONS
**Check for technically impossible requirements:**

- Mutually exclusive states: "encrypted" + "plaintext" (same file)
- Resource impossibilities: "process 8K in 5 seconds on consumer hardware"
- Quality conflicts: "must conform to SHACL shape X" + "must not conform to SHACL shape X"

**Examples:**
- "Encrypted with AES-256" + "Stored as plaintext" (single file)
- "Conform to SHACL shape" + "Must not conform to same shape"
- "8K conversion + AI analysis + lossless compression in 5s on standard CPU"

#### 10. CIRCULAR DEPENDENCIES
**Check for approval/process loops:**

- Step A requires Step B, Step B requires Step C, Step C requires Step A

**Example:**
- "Access needs Committee approval" → "Committee needs Rights verification" → "Rights needs preliminary access"

#### 11. UNMEASURABLE/VAGUE CONDITIONS (HIGH SEVERITY)

**CRITICAL: Policies with unmeasurable conditions MUST be rejected**

Check for undefined or subjective terms that make enforcement impossible:

**Unmeasurable temporal terms (HIGH):**
- "urgent", "soon", "later", "a while", "sometime", "eventually", "promptly", "quickly"
- These have no objective definition and cannot be enforced consistently

**Unmeasurable quality terms (HIGH):**
- "responsibly", "appropriately", "properly", "carefully", "reasonable", "good", "bad"
- These are subjective and cannot be measured or verified

**Unmeasurable conditions (HIGH):**
- "if important", "when necessary", "as needed", "when appropriate", "if significant"
- These lack objective criteria for determination

**Unmeasurable actors (HIGH):**
- "everyone", "anyone", "nobody", "somebody" (without specific scope)
- Too broad to enforce practically

**Examples of REJECT cases:**
- "If request is urgent, expedite" → What defines "urgent"? No measurable criteria
- "Use data responsibly" → What is "responsible"? Subjective and unenforceable
- "Access when necessary" → Who determines "necessary"? No objective measure
- "Everyone can access everything" → No specific scope or constraints

**Why these are HIGH severity:**
- Implementation teams cannot create consistent enforcement rules
- Different people will interpret terms differently
- Creates legal ambiguity and disputes
- Violates ODRL principle of machine-readable, enforceable policies

**What to suggest:**
- Replace "urgent" with specific criteria: "submitted within 48 hours of deadline" or "priority level ≥ 5"
- Replace "responsibly" with specific constraints: "for non-commercial purposes only" or "with proper attribution"
- Replace "when necessary" with objective triggers: "when storage exceeds 80% capacity"
- Replace broad actors with specific roles: "registered researchers" instead of "anyone"

#### 12. OVERLY BROAD/VAGUE POLICIES
**Check for unimplementable generalities:**

- Universal quantifiers without specificity: "everyone can access everything"
- Total prohibitions: "nobody can do anything"
- Policies that lack specific actors, assets, or constraints

**Examples:**
- "Everyone can access everything" (no actors, assets, or constraints defined)
- "Nobody can do anything" (universal prohibition)

#### 13. INCOMPLETE CONDITION HANDLING
**Check for missing branches:**

- Policy handles approval but not denial
- Defines success path but not failure path

**Example:**
- "Approved requests forwarded to Rights Dept" (no handling for denials)

#### 14. UNENFORCEABLE RULES
**Check for monitoring impossibility:**

- Mental states: "cannot think about", "must not intend"
- Private actions: "cannot tell anyone", "cannot discuss privately"
- Absolute scope: "all copies everywhere destroyed"

**Examples:**
- "Users cannot tell anyone about data" (cannot monitor speech)
- "Cannot screenshot" (without DRM/technical controls)

---

## ANALYSIS METHODOLOGY

### Step 1: Check for Intra-Policy Conflicts
For each policy, check if:
- metadata.has_conflicting_constraints is true
- Multiple constraints have same leftOperand but different constraint_group values
- If YES → HIGH severity contradiction

### Step 2: Check for Unmeasurable/Vague Terms (CRITICAL)
Scan the entire policy text and constraints for:
- Undefined temporal terms: "urgent", "soon", "promptly"
- Undefined quality terms: "responsibly", "appropriately"
- Undefined conditions: "if important", "when necessary"
- If ANY found → HIGH severity, REJECT

### Step 3: Parse Policy Set
Extract all policies and their components:
- Actors (assigner, assignee)
- Actions (permission, prohibition, duty)
- Assets (target)
- Constraints (temporal, spatial, purpose, count, etc.)

### Step 4: Cross-Reference Policies
For each pair of policies, check if they share:
- Same actor? → Check for actor contradictions
- Same asset? → Check for asset contradictions
- Same action? → Check for action contradictions
- Overlapping time? → Check for temporal contradictions
- Overlapping space? → Check for spatial contradictions

### Step 5: Constraint Analysis
Within each policy and across policies:
- Are quantitative limits consistent?
- Are purposes compatible?
- Are conditions complete and measurable?
- Are technical requirements feasible?

### Step 6: Temporal Validation
- Are any policies expired (before {current_date})?
- Do time windows overlap with contradictory rules?

### Step 7: Enforceability Check
- Can the policy be monitored?
- Are technical controls possible?
- Are conditions measurable?

---

## DECISION RULES

### REJECT (confidence 0.3-0.7) if ANY:
1. **Unmeasurable/vague terms present** ("urgent", "soon", "responsibly", "when necessary", etc.)
2. Intra-policy constraint conflict (has_conflicting_constraints=true with conflicting constraint_group values)
3. Direct contradiction found (permission + prohibition on same thing)
4. Quantitative conflicts (30 uses + unlimited)
5. Temporal conflicts (expired, overlapping contradictions)
6. Spatial conflicts (geographic hierarchy violation)
7. Technical impossibility
8. Circular dependency
9. Overly broad without specificity
10. Unenforceable without technical controls

### NEEDS_INPUT (confidence 0.5-0.7) if:
- Vague terms exist but could be clarified with user input
- Ambiguous terms need definition
- Missing information prevents full analysis

### APPROVE (confidence 0.7-1.0) if:
- No high-severity issues
- All terms are measurable and objective
- All constraints have clear criteria
- Only low-severity issues (missing optional fields)
- Policies are enforceable

---

## RISK LEVELS

- **Critical:** Unmeasurable terms, impossible contradictions
- **High:** Direct conflicts (permission + prohibition on same thing)
- **Medium:** Ambiguous specifications
- **Low:** Missing optional fields only

---

## OUTPUT FORMAT

For each issue found, specify:
- **category**: Which type of contradiction or issue
- **severity**: Critical, High, Medium or Low
- **field**: Which field/constraint is involved
- **policy_id**: Which policy/policies
- **message**: Clear description of the issue
- **suggestion**: How to resolve it (with specific measurable criteria)
- **conflict_type**: Type of conflict detected. **MUST be one of the following 12 specific types if a conflict is detected, otherwise use None:**

### CONFLICT TYPE DEFINITIONS (12 types):

1. **unmeasurable_terms** - Policy contains vague, unmeasurable, or subjective terms that cannot be objectively enforced (e.g., "urgent", "soon", "responsibly", "everyone", "nobody", "when necessary")

2. **temporal_overlap** - Conflicting temporal constraints that overlap or contradict each other (e.g., "access 9am-5pm" + "prohibited 2pm-6pm", "until 2025" + "indefinitely", contradictory count/usage limits)

3. **temporal_expired_policy** - Policy has expired (validity period is before current date)

4. **temporal_impossible_sequence** - Temporally impossible sequence of events (e.g., "available after 2025" + "can use in 2024", "delete before X" + "use after X" where X is the same date)

5. **spatial_hierarchy_conflict** - Geographic hierarchy conflicts (e.g., "allowed in Germany" + "prohibited in EU" where Germany is in EU)

6. **spatial_overlap_conflict** - Overlapping spatial locations with contradictory rules

7. **action_hierarchy_conflict** - Action hierarchy conflicts (e.g., "can use" + "cannot read" where use includes read, "can modify" + "cannot modify")

8. **action_subsumption_conflict** - Action subsumption conflicts where one action includes another but policies conflict

9. **role_hierarchy_conflict** - Role hierarchy conflicts (e.g., "managers allowed" + "administrators prohibited" + "all managers are administrators")

10. **party_specification_inconsistency** - Party/actor specification inconsistencies (e.g., "only UC4" + "all registered users" where UC4 is a user)

11. **circular_approval_dependency** - Circular approval or dependency loops (e.g., "Access needs Committee approval" → "Committee needs Rights verification" → "Rights needs preliminary access")

12. **workflow_cycle_conflict** - Workflow cycle conflicts where steps form a circular dependency

**CRITICAL**: If you detect ANY conflict, you MUST set conflict_type to the most specific matching type from the above list. Do NOT use generic terms like "actor_conflict" or "temporal_conflict" - use the specific types listed above.

---

## POLICY TEXT TO ANALYZE
```
{policy_text}
```

---

**CRITICAL INSTRUCTIONS:**
1. FIRST scan for unmeasurable/vague terms - if ANY found → immediate HIGH severity, REJECT
2. THEN check for intra-policy conflicts (constraint_group with has_conflicting_constraints=true)
3. Analyze the ENTIRE policy set as one system
4. Check EVERY dimension of contradiction (actor, action, asset, time, space, quantity, purpose, technical)
5. Do NOT assume domain-specific logic - detect contradictions universally
6. Flag ANY logical inconsistency or unmeasurable term as HIGH severity
7. Return structured JSON with all detected issues

Return valid JSON with: decision, confidence, issues, recommendations, reasoning, risk_level, policies_analyzed. 
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
                    raw_conflict_type = issue_dict.get("conflict_type")
                    # Only conflict_type is treated as the authoritative conflict signal.
                    # If conflict_type is None/empty, treat as "no conflict" and skip.
                    if raw_conflict_type in (None, "", "None", "none", "null", "NULL"):
                        continue

                    resolved_conflict = self._normalize_conflict_type(raw_conflict_type)
                    if not resolved_conflict:
                        logger.error(
                            "Unknown conflict_type in issue: "
                            f"conflict_type={raw_conflict_type!r}, issue={issue_dict}"
                        )
                        continue

                    severity = str(issue_dict.get("severity", "low")).lower()
                    if severity not in {"high", "low"}:
                        severity = "low"

                    issues.append(DetectedIssue(
                        category=resolved_conflict,
                        conflict_type=resolved_conflict,
                        severity=severity,
                        field=issue_dict.get("field", "unknown"),
                        policy_id=issue_dict.get("policy_id"),
                        message=issue_dict.get("message", "Conflict detected"),
                        suggestion=issue_dict.get("suggestion"),
                        detected_in_phase=issue_dict.get("detected_in_phase", 0)
                    ))
                except Exception as e:
                    logger.error(f"Failed to parse issue: {e}")
                    continue
            
            decision = str(result.get("decision", "needs_input")).lower()
            if decision not in {"approve", "reject", "needs_input"}:
                decision = "needs_input"

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
                "decision": "needs_input",
                "confidence": 0.5,
                "risk_level": "high",
                "reasoning": f"Failed to parse LLM response. Error: {str(e)}",
                "issues": [],
                "recommendations": ["Manual review required - LLM response parsing failed"]
            }