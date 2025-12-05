# agents/generator/generator.py (UPDATED - Universal)

"""
ODRL Generator Agent - Single LLM Call
Works with Azure OpenAI or OpenAI-compatible endpoints
Generates ODRL policies in Turtle format from approved policy text
"""

from typing import Optional
from datetime import datetime, timezone
import uuid
import re
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import logging

logger = logging.getLogger(__name__)


# ===== COMPREHENSIVE TURTLE GENERATION PROMPT =====

TURTLE_GENERATION_PROMPT = """
# ODRL POLICY GENERATOR - TURTLE FORMAT

You are an expert at generating W3C ODRL 2.2 compliant policies in Turtle (TTL) format.

**Current Date:** {current_date}

---

## INPUT

You will receive approved policy text that has passed contradiction detection. Your task is to convert it into valid ODRL Turtle.

**Policy ID:** {policy_id}

**Policy Text:**
```
{policy_text}
```

---

## TURTLE STRUCTURE REQUIREMENTS

### 1. PREFIXES (Always include these)
```turtle
@prefix odrl: <http://www.w3.org/ns/odrl/2/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix ex: <http://example.com/> .
```

---

### 2. POLICY DECLARATION
```turtle
ex:policy_{policy_id} a odrl:Policy, odrl:Set ;
    odrl:uid ex:policy_{policy_id} ;
    dct:title "Concise Policy Title"@en ;
    dct:description "Clear explanation of what this policy allows/prohibits"@en ;
    dct:creator ex:organization_creator ;
    dct:created "{current_date}T00:00:00Z"^^xsd:dateTime ;
    odrl:permission [ ... ] ;
    odrl:prohibition [ ... ] .
```

**Policy Types:**
- `odrl:Set` - General policy collection (most common)
- `odrl:Offer` - Provider offers terms to recipients
- `odrl:Agreement` - Binding agreement between parties

---

### 3. PERMISSION STRUCTURE
```turtle
odrl:permission [
    a odrl:Permission ;
    odrl:action odrl:read ;
    odrl:target ex:dataset_xyz ;
    odrl:assignee ex:party_researcher ;
    odrl:assigner ex:org_provider ;
    odrl:constraint [
        a odrl:Constraint ;
        odrl:leftOperand odrl:dateTime ;
        odrl:operator odrl:lteq ;
        odrl:rightOperand "2025-12-31"^^xsd:date ;
        rdfs:comment "Valid until end of 2025"@en ;
    ] ;
] .
```

---

### 4. ODRL ACTIONS (Only use these standard actions)

**Access Actions:**
- `odrl:read` - Read/view content
- `odrl:use` - General usage
- `odrl:index` - Index for search

**Modification Actions:**
- `odrl:modify` - Modify content
- `odrl:derive` - Create derivative works
- `odrl:reproduce` - Make copies

**Distribution Actions:**
- `odrl:distribute` - Distribute copies
- `odrl:share` - Share with others
- `odrl:sell` - Commercial distribution

**Management Actions:**
- `odrl:archive` - Archive/preserve
- `odrl:delete` - Delete content

---

### 5. CONSTRAINTS

**Temporal Constraints:**
```turtle
odrl:constraint [
    a odrl:Constraint ;
    odrl:leftOperand odrl:dateTime ;
    odrl:operator odrl:lteq ;
    odrl:rightOperand "2025-12-31"^^xsd:date ;
] .
```

**Count Constraints:**
```turtle
odrl:constraint [
    a odrl:Constraint ;
    odrl:leftOperand odrl:count ;
    odrl:operator odrl:lteq ;
    odrl:rightOperand "30"^^xsd:integer ;
] .
```

**Purpose Constraints:**
```turtle
odrl:constraint [
    a odrl:Constraint ;
    odrl:leftOperand odrl:purpose ;
    odrl:operator odrl:eq ;
    odrl:rightOperand "research" ;
] .
```

---

### 6. ODRL OPERATORS (Only use these)

**Comparison:**
- `odrl:eq` - equals
- `odrl:neq` - not equals
- `odrl:lt` - less than
- `odrl:gt` - greater than
- `odrl:lteq` - less than or equal
- `odrl:gteq` - greater than or equal

**Set Operations:**
- `odrl:isA` - instance of
- `odrl:isAnyOf` - any of these
- `odrl:isNoneOf` - none of these

---

### 7. URI CONSTRUCTION RULES

**Policy URIs:** Use `ex:policy_{policy_id}`

**Asset URIs:** Extract from policy text:
- `ex:dataset_name`
- `ex:document_name`

**Party URIs:** Extract from policy text:
- `ex:party_name`
- `ex:organization_name`

Use lowercase, replace spaces with underscores.

---

### 8. METADATA REQUIREMENTS
```turtle
dct:title "Research Access to Medieval Manuscripts"@en ;
dct:description "UC4 partners may read and use the medieval manuscript collection for educational and research purposes until December 31, 2025"@en ;
```

---

## OUTPUT REQUIREMENTS

1. **Return ONLY valid Turtle syntax**
2. **NO markdown code blocks** (no ```)
3. **NO explanatory text**
4. **Start with @prefix declarations**
5. **Use the provided policy_id: {policy_id}**
6. **Include dct:title and dct:description**
7. **Use proper datatypes** (^^xsd:date, ^^xsd:integer)

---

Generate the ODRL Turtle policy now:
"""


# ===== GENERATOR CLASS =====

class Generator:
    """
    ODRL Generator Agent - Single LLM Call
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
        Initialize Generator with either Azure or OpenAI-compatible endpoint
        
        Azure usage:
            Generator(
                api_key="...",
                api_version="2024-10-01-preview",
                azure_endpoint="https://....openai.azure.com/",
                model="gpt-4o-2024-11-20"
            )
        
        Local/Fraunhofer usage:
            Generator(
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
    
    def generate(self, policy_text: str, policy_id: Optional[str] = None) -> dict:
        """
        Generate ODRL Turtle from approved policy text
        
        Args:
            policy_text: Natural language policy description (approved by reasoner)
            policy_id: Optional policy ID (generated if not provided)
            
        Returns:
            Dict with 'odrl_turtle' and metadata
        """
        
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if not policy_id:
            policy_id = uuid.uuid4().hex[:8]
        
        logger.info("=" * 60)
        logger.info(f"STARTING ODRL GENERATION ({self.endpoint_type})")
        logger.info(f"Policy ID: {policy_id}")
        logger.info(f"Input length: {len(policy_text)} characters")
        logger.info("=" * 60)
        
        # Format prompt
        prompt = TURTLE_GENERATION_PROMPT.format(
            current_date=current_date,
            policy_text=policy_text,
            policy_id=policy_id
        )
        
        # Single LLM call
        logger.info("[LLM] Generating ODRL Turtle...")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse response
        odrl_turtle = self._clean_turtle(response.content)
        
        logger.info("=" * 60)
        logger.info("GENERATION COMPLETE")
        logger.info(f"Turtle length: {len(odrl_turtle)} characters")
        logger.info(f"Triples (approx): {odrl_turtle.count(';') + odrl_turtle.count('.')}")
        logger.info("=" * 60)
        
        return {
            "odrl_turtle": odrl_turtle,
            "format": "turtle",
            "policy_id": policy_id,
            "generated_at": current_date
        }
    
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