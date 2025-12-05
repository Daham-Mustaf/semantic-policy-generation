# agents/validator/odrl_validation_tool.py
"""
ODRL Validation Tool - Pure SHACL Validation (No LLM)
Validates ODRL Turtle against SHACL shapes
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Set, Optional
from enum import Enum
from abc import ABC, abstractmethod
import rdflib
from pyshacl import validate

import logging
logger = logging.getLogger(__name__)

# ============================================
# 1. ValidationIssue - Single SHACL violation
# ============================================

@dataclass
class ValidationIssue:
    """Single SHACL violation detected"""
    issue_type: str              # e.g., "Missing Policy UID"
    focus_node: str              # e.g., "ex:policy_123"
    property_path: str           # e.g., "odrl:uid"
    actual_value: str            # e.g., "not specified"
    constraint_violated: str     # Human-readable explanation
    severity: str = "Violation"  # "Violation" or "Warning"


# ============================================
# 2. ValidationReport - Complete validation result
# ============================================

@dataclass
class ValidationReport:
    """Complete validation report for LLM feedback"""
    user_text: str          # Original policy text from user
    generated_kg: str       # Generated ODRL Turtle
    is_valid: bool          # True if no violations
    issues: List[ValidationIssue]  # List of all violations found
    
    def to_learning_prompt(self) -> str:
        """
        Generate structured prompt for LLM regeneration
        Shows: user intent + generated ODRL + specific violations
        """
        lines = []
        
        # Header
        lines.append("# ODRL Knowledge Graph Validation Report")
        lines.append("")
        
        # Show what user wanted
        lines.append("## Original User Request")
        lines.append(f'"{self.user_text}"')
        lines.append("")
        
        # Show what was generated
        lines.append("## Generated Knowledge Graph")
        lines.append("```turtle")
        lines.append(self.generated_kg.strip())
        lines.append("```")
        lines.append("")
        
        # Show validation results
        lines.append("## Validation Results")
        if self.is_valid:
            lines.append("**Status**: VALID ")
            lines.append("")
            lines.append("The generated knowledge graph conforms to all ODRL validation rules.")
        else:
            lines.append(f"**Status**: INVALID  - {len(self.issues)} issue(s) detected")
            lines.append("")
            
            # Group issues by type
            issue_groups = self._group_issues_by_type()
            
            for issue_type, type_issues in issue_groups.items():
                lines.append(f"### {issue_type}")
                for i, issue in enumerate(type_issues, 1):
                    lines.append(f"{i}. **Node**: `{issue.focus_node}`")
                    lines.append(f"   **Property**: `{issue.property_path}`")
                    lines.append(f"   **Current Value**: `{issue.actual_value}`")
                    lines.append(f"   **Constraint Violated**: {issue.constraint_violated}")
                    if issue.severity != "Violation":
                        lines.append(f"   **Severity**: {issue.severity}")
                    lines.append("")
            
            # Learning notes
            lines.append("## Learning Notes")
            lines.append("The above issues indicate where the knowledge graph doesn't conform to ODRL standards.")
            lines.append("Review the constraint violations to understand what corrections are needed.")
        
        return "\n".join(lines)
    
    def _group_issues_by_type(self) -> Dict[str, List[ValidationIssue]]:
        """Group issues by type for better organization"""
        groups = {}
        for issue in self.issues:
            if issue.issue_type not in groups:
                groups[issue.issue_type] = []
            groups[issue.issue_type].append(issue)
        return groups


# ============================================
# 3. ODRL Vocabulary Definitions
# ============================================

class OperatorType(Enum):
    """ODRL Core Constraint Operators"""
    EQ = "eq"
    GT = "gt"
    GTEQ = "gteq"
    LT = "lt"
    LTEQ = "lteq"
    NEQ = "neq"
    IS_A = "isA"
    HAS_PART = "hasPart"
    IS_PART_OF = "isPartOf"
    IS_ALL_OF = "isAllOf"
    IS_ANY_OF = "isAnyOf"
    IS_NONE_OF = "isNoneOf"


@dataclass
class LeftOperandInfo:
    """Information about ODRL left operands"""
    uri: str
    label: str
    definition: str
    compatible_operators: Set[OperatorType]
    expected_datatype: Optional[str] = None


class ODRLLeftOperands:
    """Registry of ODRL left operands"""
    
    OPERANDS = {
        "dateTime": LeftOperandInfo(
            uri="http://www.w3.org/ns/odrl/2/dateTime",
            label="Datetime",
            definition="The date (and optional time) of exercising the action",
            compatible_operators={OperatorType.LT, OperatorType.LTEQ, OperatorType.GT, OperatorType.GTEQ, OperatorType.EQ},
            expected_datatype="xsd:date"
        ),
        "count": LeftOperandInfo(
            uri="http://www.w3.org/ns/odrl/2/count",
            label="Count",
            definition="Numeric count of executions",
            compatible_operators={OperatorType.LT, OperatorType.LTEQ, OperatorType.GT, OperatorType.GTEQ, OperatorType.EQ},
            expected_datatype="xsd:integer"
        ),
        "elapsedTime": LeftOperandInfo(
            uri="http://www.w3.org/ns/odrl/2/elapsedTime",
            label="Elapsed Time",
            definition="A continuous elapsed time period",
            compatible_operators={OperatorType.EQ, OperatorType.LT, OperatorType.LTEQ},
            expected_datatype="xsd:duration"
        ),
        "payAmount": LeftOperandInfo(
            uri="http://www.w3.org/ns/odrl/2/payAmount",
            label="Payment Amount",
            definition="The amount of a financial payment",
            compatible_operators={OperatorType.EQ, OperatorType.LT, OperatorType.LTEQ, OperatorType.GT, OperatorType.GTEQ},
            expected_datatype="xsd:decimal"
        ),
        "percentage": LeftOperandInfo(
            uri="http://www.w3.org/ns/odrl/2/percentage",
            label="Asset Percentage",
            definition="A percentage amount of the target Asset",
            compatible_operators={OperatorType.EQ, OperatorType.LT, OperatorType.LTEQ, OperatorType.GT, OperatorType.GTEQ},
            expected_datatype="xsd:decimal"
        ),
        "spatial": LeftOperandInfo(
            uri="http://www.w3.org/ns/odrl/2/spatial",
            label="Geospatial Named Area",
            definition="A named geospatial area",
            compatible_operators={OperatorType.EQ, OperatorType.IS_A, OperatorType.IS_ANY_OF, OperatorType.IS_NONE_OF}
        ),
        "purpose": LeftOperandInfo(
            uri="http://www.w3.org/ns/odrl/2/purpose",
            label="Purpose",
            definition="A defined purpose for exercising the action",
            compatible_operators={OperatorType.EQ, OperatorType.IS_A, OperatorType.IS_ANY_OF, OperatorType.IS_NONE_OF}
        ),
        "recipient": LeftOperandInfo(
            uri="http://www.w3.org/ns/odrl/2/recipient",
            label="Recipient",
            definition="The party receiving the result",
            compatible_operators={OperatorType.EQ, OperatorType.IS_A, OperatorType.IS_ANY_OF, OperatorType.IS_NONE_OF}
        ),
    }
    
    @classmethod
    def get_operand(cls, name: str) -> Optional[LeftOperandInfo]:
        return cls.OPERANDS.get(name)
    
    @classmethod
    def list_operands(cls) -> List[str]:
        return list(cls.OPERANDS.keys())


# ============================================
# 4. Base Validator (Abstract)
# ============================================

class BaseValidator(ABC):
    """Abstract base for SHACL validators"""
    
    @abstractmethod
    def get_shape_ttl(self) -> str:
        """Return SHACL shape in Turtle format"""
        pass
    
    @abstractmethod
    def process_violations(self, violations: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Convert SHACL violations to ValidationIssues"""
        pass


# ============================================
# 5. Concrete Validators
# ============================================

class PolicyStructureValidator(BaseValidator):
    """Validates basic ODRL policy structure"""
    
    def get_shape_ttl(self) -> str:
        return """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .

<PolicyStructureShape> a sh:NodeShape ;
    sh:targetClass odrl:Policy ;
    sh:property [
        sh:path odrl:uid ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:nodeKind sh:IRI ;
        sh:message "Policy must have exactly one uid with IRI value" ;
    ] ;
    sh:or (
        [ sh:property [ sh:path odrl:permission ; sh:minCount 1 ] ]
        [ sh:property [ sh:path odrl:prohibition ; sh:minCount 1 ] ]
        [ sh:property [ sh:path odrl:obligation ; sh:minCount 1 ] ]
    ) ;
    sh:message "Policy must have at least one rule" .
"""
    
    def process_violations(self, violations: List[Dict[str, Any]]) -> List[ValidationIssue]:
        issues = []
        for violation in violations:
            constraint_type = violation.get("source_constraint_component", "")
            
            if "uid" in str(violation.get("result_path", "")):
                issue_type = "Missing Policy UID"
                constraint_violated = "ODRL Policy must have exactly one odrl:uid property with an IRI value"
            elif "OrConstraintComponent" in constraint_type:
                issue_type = "Missing Policy Rules"
                constraint_violated = "ODRL Policy must contain at least one Rule (permission/prohibition/obligation)"
            else:
                issue_type = "Policy Structure Error"
                constraint_violated = violation.get("message", "Unknown policy structure violation")
            
            issues.append(ValidationIssue(
                issue_type=issue_type,
                focus_node=str(violation.get("focus_node", "")),
                property_path=str(violation.get("result_path", "")),
                actual_value=str(violation.get("value", "not specified")),
                constraint_violated=constraint_violated
            ))
        
        return issues


class ConstraintStructureValidator(BaseValidator):
    """Validates basic constraint structure"""
    
    def get_shape_ttl(self) -> str:
        left_operands = " ".join([f"odrl:{op}" for op in ODRLLeftOperands.list_operands()])
        operators = " ".join([f"odrl:{op.value}" for op in OperatorType])
        
        return f"""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .

<ConstraintStructureShape> a sh:NodeShape ;
    sh:targetClass odrl:Constraint ;
    
    sh:property [
        sh:path odrl:leftOperand ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:in ( {left_operands} ) ;
        sh:message "Invalid or missing left operand" ;
    ] ;
    
    sh:property [
        sh:path odrl:operator ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:in ( {operators} ) ;
        sh:message "Invalid or missing operator" ;
    ] ;
    
    sh:xone (
        [ sh:property [ sh:path odrl:rightOperand ; sh:minCount 1 ] ]
        [ sh:property [ sh:path odrl:rightOperandReference ; sh:minCount 1 ] ]
    ) ;
    sh:message "Missing right operand or reference" .
"""
    
    def process_violations(self, violations: List[Dict[str, Any]]) -> List[ValidationIssue]:
        issues = []
        for violation in violations:
            constraint_type = violation.get("source_constraint_component", "")
            
            if "leftOperand" in str(violation.get("result_path", "")):
                issue_type = "Invalid Left Operand"
                actual_operand = self._extract_from_uri(violation.get("value", ""))
                constraint_violated = f"Left operand '{actual_operand}' is not in ODRL Core. Valid: {', '.join(ODRLLeftOperands.list_operands())}"
            elif "operator" in str(violation.get("result_path", "")):
                issue_type = "Invalid Operator"
                actual_op = self._extract_from_uri(violation.get("value", ""))
                valid_ops = ', '.join([op.value for op in OperatorType])
                constraint_violated = f"Operator '{actual_op}' is not in ODRL Core. Valid: {valid_ops}"
            elif "XoneConstraintComponent" in constraint_type:
                issue_type = "Missing Right Operand"
                constraint_violated = "Must have either rightOperand or rightOperandReference"
            else:
                issue_type = "Constraint Structure Error"
                constraint_violated = violation.get("message", "Unknown constraint violation")
            
            issues.append(ValidationIssue(
                issue_type=issue_type,
                focus_node=str(violation.get("focus_node", "")),
                property_path=str(violation.get("result_path", "")),
                actual_value=str(violation.get("value", "not specified")),
                constraint_violated=constraint_violated
            ))
        
        return issues
    
    def _extract_from_uri(self, uri_value: str) -> str:
        """Extract name from URI"""
        if "odrl/2/" in uri_value:
            return uri_value.split("odrl/2/")[-1]
        elif ":" in uri_value:
            return uri_value.split(":")[-1]
        return uri_value


class ConstraintCompatibilityValidator(BaseValidator):
    """Validates operand-operator compatibility"""
    
    def get_shape_ttl(self) -> str:
        rules = []
        for operand_name, operand_info in ODRLLeftOperands.OPERANDS.items():
            compatible_ops = " ".join([f"odrl:{op.value}" for op in operand_info.compatible_operators])
            
            #FIXED: Proper SPARQL syntax with PREFIX declaration
            rule = f"""
<CompatibilityRule_{operand_name}> a sh:NodeShape ;
    sh:targetClass odrl:Constraint ;
    sh:sparql [
        sh:select '''
            PREFIX odrl: <http://www.w3.org/ns/odrl/2/>
            SELECT $this ?operator
            WHERE {{
                $this odrl:leftOperand odrl:{operand_name} .
                $this odrl:operator ?operator .
                FILTER (?operator NOT IN ({compatible_ops}))
            }}
        ''' ;
        sh:message "Incompatible operator for {operand_name}" ;
        sh:severity sh:Warning ;
    ] .
"""
            rules.append(rule)
        
        return f"""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .

{chr(10).join(rules)}
"""
    
    def process_violations(self, violations: List[Dict[str, Any]]) -> List[ValidationIssue]:
        issues = []
        for violation in violations:
            focus_node = str(violation.get("focus_node", ""))
            operator_value = str(violation.get("value", ""))
            message = violation.get("message", "")
            
            # Parse operand from message
            operand_name = None
            if "for " in message:
                operand_name = message.split("for ")[-1].strip()
            
            if operand_name:
                operand_info = ODRLLeftOperands.get_operand(operand_name)
                if operand_info:
                    valid_ops = ', '.join([op.value for op in operand_info.compatible_operators])
                    operator_short = self._extract_from_uri(operator_value)
                    constraint_violated = f"Operator '{operator_short}' not compatible with leftOperand '{operand_name}'. Valid: {valid_ops}"
                else:
                    constraint_violated = f"Unknown operand '{operand_name}'"
            else:
                constraint_violated = "Operator-operand compatibility issue"
            
            issues.append(ValidationIssue(
                issue_type="Operator Compatibility",
                focus_node=focus_node,
                property_path="odrl:operator",
                actual_value=operator_value,
                constraint_violated=constraint_violated,
                severity="Warning"
            ))
        
        return issues
    
    def _extract_from_uri(self, uri_value: str) -> str:
        """Extract name from URI"""
        if "odrl/2/" in uri_value:
            return uri_value.split("odrl/2/")[-1]
        elif "#" in uri_value:
            return uri_value.split("#")[-1]
        elif "/" in uri_value:
            return uri_value.split("/")[-1]
        return uri_value


# ============================================
# 6. Main Validation Tool
# ============================================

class ODRLValidationTool:
    """
    Main validation tool for ODRL knowledge graphs
    Uses multiple SHACL validators to check compliance
    """
    
    def __init__(self):
        self.validators = [
            PolicyStructureValidator(),
            ConstraintStructureValidator(),
            # ConstraintCompatibilityValidator(),
        ]
    
    def validate_kg(self, user_text: str, kg_turtle: str) -> ValidationReport:
        """
        Validate ODRL Turtle and return structured report
        
        Args:
            user_text: Original user policy text
            kg_turtle: Generated ODRL Turtle
            
        Returns:
            ValidationReport with all issues found
        """
        all_issues = []
        
        # Run each validator
        for validator in self.validators:
            violations = self._run_shacl_validation(kg_turtle, validator.get_shape_ttl())
            issues = validator.process_violations(violations)
            all_issues.extend(issues)
        
        is_valid = len(all_issues) == 0
        
        return ValidationReport(
            user_text=user_text,
            generated_kg=kg_turtle,
            is_valid=is_valid,
            issues=all_issues
        )
    
    def _run_shacl_validation(self, data_ttl: str, shape_ttl: str) -> List[Dict[str, Any]]:
        """Run SHACL validation and extract violations"""
        try:
            # Parse data and shape graphs
            data_graph = rdflib.Graph()
            data_graph.parse(data=data_ttl, format="turtle")
            
            shape_graph = rdflib.Graph()
            shape_graph.parse(data=shape_ttl, format="turtle")
            
            # Run validation
            conforms, report_graph, report_text = validate(
                data_graph,
                shacl_graph=shape_graph,
                inference='rdfs',
                serialize_report_graph=True,
                debug=False
            )
            
            # Handle report graph
            if isinstance(report_graph, bytes):
                parsed_report_graph = rdflib.Graph()
                parsed_report_graph.parse(data=report_graph.decode('utf-8'), format="turtle")
                report_graph = parsed_report_graph
            
            # Extract violations from report
            SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
            violations = []
            
            for s in report_graph.subjects(rdflib.RDF.type, SH.ValidationResult):
                violation = {}
                for pred, obj in report_graph.predicate_objects(s):
                    if pred == SH.focusNode:
                        violation["focus_node"] = str(obj)
                    elif pred == SH.value:
                        violation["value"] = str(obj)
                    elif pred == SH.sourceConstraintComponent:
                        violation["source_constraint_component"] = str(obj)
                    elif pred == SH.resultPath:
                        violation["result_path"] = str(obj)
                    elif pred == SH.resultMessage:
                        violation["message"] = str(obj)
                
                violations.append(violation)
            
            return violations
            
        except Exception as e:
            logger.error(f"SHACL validation error: {e}")
            return [{
                "message": f"Validation error: {str(e)}",
                "focus_node": "",
                "value": "",
                "source_constraint_component": "",
                "result_path": ""
            }]


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    user_text = "Alice can read the document until December 31, 2025"
    
    # Example with errors
    kg_with_errors = """
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .
@prefix ex: <http://example.com/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:policy1 a odrl:Policy ;
    odrl:permission [
        a odrl:Permission ;
        odrl:action odrl:read ;
        odrl:target ex:document ;
        odrl:constraint [
            a odrl:Constraint ;
            odrl:leftOperand odrl:dateTime ;
            odrl:operator odrl:invalidOp ;
            odrl:rightOperand "2025-12-31"^^xsd:date ;
        ] ;
    ] .
"""
    
    # Run validation
    tool = ODRLValidationTool()
    report = tool.validate_kg(user_text, kg_with_errors)
    
    # Print learning prompt
    print(report.to_learning_prompt())