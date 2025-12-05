# agents/validator/validator_agent.py

"""
ODRL Validator Agent - Single LLM Call for Regeneration
Works with Azure OpenAI or OpenAI-compatible endpoints
Validates ODRL Turtle against SHACL shapes and provides feedback for fixing
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import re
import logging
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agents.validator.odrl_validation_tool import ODRLValidationTool, ValidationReport

logger = logging.getLogger(__name__)


# ===== REGENERATION PROMPT =====

ODRL_REGENERATION_PROMPT = """
# ODRL SHACL VIOLATION FIXER

You are an expert at fixing W3C ODRL 2.2 compliance issues in Turtle format.

**Current Date:** {current_date}

---

{validation_learning_prompt}

---

## FIXING GUIDELINES

### 1. PRESERVE POLICY MEANING
- Do NOT change who can do what
- Do NOT change which resources
- Do NOT change time periods, counts, or purposes
- Do NOT remove constraints
- ONLY fix technical SHACL compliance issues

### 2. COMMON FIXES

**Missing odrl:uid:**
```turtle
# WRONG
ex:policy_123 a odrl:Policy ;
    odrl:permission [ ... ] .

# CORRECT
ex:policy_123 a odrl:Policy ;
    odrl:uid ex:policy_123 ;  # ← ADD THIS (must match policy URI)
    odrl:permission [ ... ] .
```

**Invalid Operator:**
```turtle
# WRONG
odrl:operator "lte" .          # Missing odrl: prefix
odrl:operator invalidOp .      # Not in ODRL vocabulary

# CORRECT
odrl:operator odrl:lteq .      # Use odrl: prefix and valid operator
```

**Valid ODRL Operators:**
- `odrl:eq` (equals)
- `odrl:neq` (not equals)
- `odrl:lt` (less than)
- `odrl:lteq` (less than or equal)
- `odrl:gt` (greater than)
- `odrl:gteq` (greater than or equal)
- `odrl:isA` (instance of)
- `odrl:isAnyOf` (any of set)
- `odrl:isNoneOf` (none of set)

**Invalid leftOperand:**
```turtle
# WRONG
odrl:leftOperand "dateTime" .  # Missing odrl: prefix
odrl:leftOperand count .       # Missing odrl: prefix

# CORRECT
odrl:leftOperand odrl:dateTime .
odrl:leftOperand odrl:count .
```

**Valid ODRL leftOperands:**
- `odrl:dateTime` (temporal constraints)
- `odrl:count` (usage count)
- `odrl:purpose` (purpose restrictions)
- `odrl:spatial` (geographic restrictions)
- `odrl:recipient` (recipient restrictions)
- `odrl:elapsedTime` (duration constraints)
- `odrl:payAmount` (payment amounts)
- `odrl:percentage` (percentage amounts)

**Missing Constraint Type:**
```turtle
# WRONG
odrl:constraint [
    odrl:leftOperand odrl:dateTime ;  # Missing type declaration
    odrl:operator odrl:lteq ;
    odrl:rightOperand "2025-12-31"^^xsd:date ;
] .

# CORRECT
odrl:constraint [
    a odrl:Constraint ;  # ← ADD THIS
    odrl:leftOperand odrl:dateTime ;
    odrl:operator odrl:lteq ;
    odrl:rightOperand "2025-12-31"^^xsd:date ;
] .
```

**Missing Permission/Prohibition Type:**
```turtle
# WRONG
odrl:permission [
    odrl:action odrl:read ;  # Missing type declaration
    odrl:target ex:dataset ;
] .

# CORRECT
odrl:permission [
    a odrl:Permission ;  # ← ADD THIS
    odrl:action odrl:read ;
    odrl:target ex:dataset ;
] .
```

**Wrong Datatype:**
```turtle
# WRONG
odrl:rightOperand "2025-12-31" .      # Missing ^^xsd:date
odrl:rightOperand "30" .              # Missing ^^xsd:integer

# CORRECT
odrl:rightOperand "2025-12-31"^^xsd:date .
odrl:rightOperand "30"^^xsd:integer .
```

---

## OUTPUT REQUIREMENTS

1. **Return ONLY the corrected Turtle**
2. **NO markdown code blocks** (no ```)
3. **NO explanatory text**
4. **Start with @prefix declarations**
5. **Preserve all metadata** (dct:title, dct:description, rdfs:comment)
6. **Fix ONLY the specific SHACL violations listed above**
7. **Do NOT change the policy meaning**

---

Return the corrected ODRL Turtle now:
"""


# ===== VALIDATOR AGENT CLASS =====

class ValidatorAgent:
    """
    ODRL Validator Agent
    Works with Azure OpenAI or OpenAI-compatible endpoints
    Validates ODRL against SHACL and provides regeneration guidance
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
        Initialize ValidatorAgent with either Azure or OpenAI-compatible endpoint
        
        Azure usage:
            ValidatorAgent(
                api_key="...",
                api_version="2024-10-01-preview",
                azure_endpoint="https://....openai.azure.com/",
                model="gpt-4o-2024-11-20"
            )
        
        Local/Fraunhofer usage:
            ValidatorAgent(
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
        
        self.validator_tool = ODRLValidationTool()
    
    def validate(
        self, 
        policy_text: str, 
        odrl_turtle: str
    ) -> Dict[str, Any]:
        """
        Validate ODRL Turtle against SHACL shapes
        
        Args:
            policy_text: Original natural language policy
            odrl_turtle: Generated ODRL in Turtle format
            
        Returns:
            Dict with validation results and ValidationReport
        """
        
        logger.info("=" * 60)
        logger.info(f"STARTING SHACL VALIDATION ({self.endpoint_type})")
        logger.info(f"Policy text length: {len(policy_text)} characters")
        logger.info(f"ODRL Turtle length: {len(odrl_turtle)} characters")
        logger.info("=" * 60)
        
        # Run SHACL validation
        validation_report = self.validator_tool.validate_kg(policy_text, odrl_turtle)
        
        logger.info(f"Validation complete: {'VALID' if validation_report.is_valid else 'INVALID'}")
        logger.info(f"Issues found: {len(validation_report.issues)}")
        
        if validation_report.issues:
            logger.info("Issue breakdown:")
            issue_types = {}
            for issue in validation_report.issues:
                issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
            
            for issue_type, count in issue_types.items():
                logger.info(f"  - {issue_type}: {count}")
        
        return {
            "is_valid": validation_report.is_valid,
            "validation_report": validation_report
        }
    
    def regenerate(
        self,
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Regenerate ODRL by fixing SHACL violations
        
        Args:
            validation_result: Result from validate() containing ValidationReport
            
        Returns:
            Dict with corrected ODRL
        """
        
        validation_report: ValidationReport = validation_result["validation_report"]
        
        if validation_report.is_valid:
            logger.info("ODRL is already valid - no regeneration needed")
            return {
                "odrl_turtle": validation_report.generated_kg,
                "regenerated": False
            }
        
        logger.info("=" * 60)
        logger.info("STARTING ODRL REGENERATION")
        logger.info(f"Fixing {len(validation_report.issues)} issues")
        logger.info("=" * 60)
        
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Use the learning prompt from ValidationReport
        learning_prompt = validation_report.to_learning_prompt()
        
        # Create full regeneration prompt
        prompt = ODRL_REGENERATION_PROMPT.format(
            current_date=current_date,
            validation_learning_prompt=learning_prompt
        )
        
        # Single LLM call to fix
        logger.info("[LLM] Invoking regeneration...")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Clean response
        corrected_turtle = self._clean_turtle(response.content)
        
        logger.info("=" * 60)
        logger.info("REGENERATION COMPLETE")
        logger.info(f"Corrected Turtle length: {len(corrected_turtle)} characters")
        logger.info("=" * 60)
        
        return {
            "odrl_turtle": corrected_turtle,
            "regenerated": True,
            "original_issues": [
                {
                    "issue_type": issue.issue_type,
                    "focus_node": issue.focus_node,
                    "property_path": issue.property_path,
                    "actual_value": issue.actual_value,
                    "constraint_violated": issue.constraint_violated,
                    "severity": issue.severity
                }
                for issue in validation_report.issues
            ]
        }
    
    def validate_and_regenerate(
        self,
        policy_text: str,
        odrl_turtle: str,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Validate and regenerate until valid or max attempts reached
        
        Args:
            policy_text: Original natural language policy
            odrl_turtle: Generated ODRL in Turtle format
            max_attempts: Maximum regeneration attempts
            
        Returns:
            Dict with final ODRL and validation status
        """
        
        logger.info("=" * 60)
        logger.info("STARTING VALIDATE-AND-REGENERATE LOOP")
        logger.info(f"Max attempts: {max_attempts}")
        logger.info("=" * 60)
        
        attempt = 1
        all_attempts = []
        
        while attempt <= max_attempts:
            logger.info(f"\n{'='*60}")
            logger.info(f"ATTEMPT {attempt}/{max_attempts}")
            logger.info(f"{'='*60}")
            
            # Validate
            validation_result = self.validate(policy_text, odrl_turtle)
            validation_report = validation_result["validation_report"]
            
            attempt_info = {
                "attempt": attempt,
                "is_valid": validation_report.is_valid,
                "issues": [
                    {
                        "issue_type": issue.issue_type,
                        "focus_node": issue.focus_node,
                        "property_path": issue.property_path,
                        "actual_value": issue.actual_value,
                        "constraint_violated": issue.constraint_violated,
                        "severity": issue.severity
                    }
                    for issue in validation_report.issues
                ],
                "odrl_turtle": odrl_turtle
            }
            
            if validation_report.is_valid:
                logger.info(f" VALIDATION PASSED on attempt {attempt}")
                all_attempts.append(attempt_info)
                
                return {
                    "success": True,
                    "final_odrl": odrl_turtle,
                    "attempts": attempt,
                    "all_attempts": all_attempts
                }
            
            logger.info(f" VALIDATION FAILED - {len(validation_report.issues)} issues")
            
            # Last attempt - don't regenerate
            if attempt == max_attempts:
                logger.info(f"Max attempts ({max_attempts}) reached")
                all_attempts.append(attempt_info)
                break
            
            # Regenerate
            regen_result = self.regenerate(validation_result)
            odrl_turtle = regen_result["odrl_turtle"]
            
            all_attempts.append(attempt_info)
            attempt += 1
        
        # Failed after max attempts
        logger.info("=" * 60)
        logger.info(f"FAILED - Could not fix issues after {max_attempts} attempts")
        logger.info("=" * 60)
        
        return {
            "success": False,
            "final_odrl": odrl_turtle,
            "attempts": max_attempts,
            "all_attempts": all_attempts,
            "final_issues": attempt_info["issues"]
        }
    
    # ===== HELPER METHODS =====
    
    def _clean_turtle(self, content: str) -> str:
        """Remove markdown code blocks and normalize whitespace"""
        
        # Remove markdown code blocks
        content = re.sub(r'```turtle\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        # Remove any explanatory text before first @prefix
        lines = content.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('@prefix'):
                start_idx = i
                break
        
        content = '\n'.join(lines[start_idx:])
        
        # Normalize whitespace
        content = content.strip()
        
        return content