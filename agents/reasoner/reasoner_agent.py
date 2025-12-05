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
# ODRL POLICY CONFLICT DETECTOR

**Current Date:** {current_date}

You are an expert at detecting contradictions in ODRL policies. Analyze the policy text systematically through 6 phases, then make a final decision.

---

## PHASE 1: CRITICAL VAGUENESS DETECTION (CHECK FIRST!)

### Unmeasurable Terms (HIGH SEVERITY - IMMEDIATE REJECT)

Scan for these terms that make enforcement impossible:

**Temporal:** "urgent", "soon", "later", "promptly", "quickly", "eventually"
**Quality:** "responsibly", "appropriately", "properly", "carefully", "reasonable"  
**Conditional:** "if important", "when necessary", "as needed", "if significant"
**Actor:** "everyone", "anyone", "nobody" (without scope)
**Overly Broad:** "Everyone can access everything", "Nobody can do anything"

**Action:** If ANY found → Mark as Phase 1 issue, set decision to REJECT

---

## PHASE 2: TEMPORAL CONFLICTS

Check for:
1. **Expired policies**: Any end_date before {current_date}
2. **Overlapping contradictions**: "9am-5pm allowed" + "2pm-6pm prohibited" = conflict at 2pm-5pm
3. **Impossible sequences**: "Available after 2025" + "Use in 2024" = impossible

**Resolution:** Specific-over-general, prohibit-on-ambiguity

---

## PHASE 3: SPATIAL CONFLICTS

Check for geographic hierarchy violations:
- "Permitted in Germany" + "Prohibited in EU" = CONFLICT (Germany ⊂ EU)
- "Allowed in Berlin" + "Prohibited in Germany" = CONFLICT (Berlin ⊂ Germany)

**Resolution:** Narrower scope takes precedence

---

## PHASE 4: ACTION HIERARCHY CONFLICTS

ODRL Action Hierarchy:
```
use
├── reproduce
├── distribute
│   └── share (share ⊑ distribute)
├── modify
│   └── adapt
└── read
```

Check for:
- "Can share" + "Cannot distribute" = CONFLICT (share is a type of distribute)
- "Can use" + "Cannot read" = CONFLICT (read is part of use)
- Same action with both permission AND prohibition

**Resolution:** Prohibition wins

---

## PHASE 5: ROLE & PARTY CONFLICTS

Check for:
- "Managers required" + "Administrators prohibited" + "Managers ⊂ Administrators" = IMPOSSIBLE
- "Only UC4" + "All partners" (if UC4 is partner) = AMBIGUOUS
- Same actor with both permission AND prohibition for same action/target

**Resolution:** Apply role hierarchy

---

## PHASE 6: CIRCULAR DEPENDENCIES

Check for approval loops:
- Access → Committee approval → Rights verification → Access = CYCLE

**Resolution:** Break at weakest link

---

## DECISION RULES

**REJECT** if ANY:
- Unmeasurable/vague terms (Phase 1)
- High severity conflicts
- Expired policies
- Circular dependencies
- Impossible contradictions
- Same actor/action/target with both permission AND prohibition

**NEEDS_INPUT** if:
- Ambiguous terms need clarification
- Missing information prevents analysis
- Role/geographic hierarchy unclear

**APPROVE** if:
- No high-severity issues
- All constraints measurable
- Policies enforceable
- No contradictions

---

## RISK LEVELS

- **Critical:** Unmeasurable terms, impossible contradictions
- **High:** Direct conflicts (permission + prohibition on same thing)
- **Medium:** Ambiguous specifications
- **Low:** Missing optional fields only

---

## OUTPUT FORMAT

Return ONLY valid JSON (no markdown, no backticks):

{{
  "decision": "approve|reject|needs_input",
  "confidence": 0.85,
  "risk_level": "critical|high|medium|low",
  "reasoning": "Comprehensive explanation covering all 6 phases",
  "issues": [
    {{
      "category": "unmeasurable_terms",
      "severity": "high",
      "field": "constraint.purpose",
      "policy_id": "implicit_policy_1",
      "message": "Term 'urgent' is unmeasurable",
      "suggestion": "Replace 'urgent' with 'priority >= 5' or 'within 48 hours'",
      "detected_in_phase": 1
    }}
  ],
  "recommendations": [
    "Specific actionable recommendation 1"
  ]
}}

---

## POLICY TEXT TO ANALYZE
```
{policy_text}
```

---

**IMPORTANT:**
- Return ONLY the JSON object, no other text
- Work systematically through phases 1-6
- Be specific about which phase detected each issue
- Always provide concrete suggestions for fixing issues

Return your analysis now:
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
        logger.info(f"Confidence: {result['confidence']:.0%}")
        logger.info(f"Risk Level: {result['risk_level'].upper()}")
        logger.info(f"Total Issues: {len(result['issues'])}")
        logger.info("=" * 60)
        
        return result
    
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
                    issues.append(DetectedIssue(
                        category=ConflictType(issue_dict["category"]),
                        severity=issue_dict["severity"],
                        field=issue_dict.get("field", "unknown"),
                        policy_id=issue_dict.get("policy_id"),
                        message=issue_dict["message"],
                        suggestion=issue_dict.get("suggestion"),
                        detected_in_phase=issue_dict.get("detected_in_phase", 0)
                    ))
                except Exception as e:
                    logger.error(f"Failed to parse issue: {e}")
                    continue
            
            return {
                "decision": result.get("decision", "needs_input"),
                "confidence": result.get("confidence", 0.5),
                "risk_level": result.get("risk_level", "medium"),
                "reasoning": result.get("reasoning", ""),
                "issues": [issue.model_dump() for issue in issues],
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