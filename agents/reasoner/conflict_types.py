
"""
Conflict Type Taxonomy for ODRL Policy Reasoning
Based on W3C ODRL semantics and production data space observations
"""

from enum import Enum
from typing import List, Dict, Any, Set
from pydantic import BaseModel

class ConflictType(str, Enum):
    """Six primary conflict categories in natural language policies"""
    
    # Category 1: Vagueness & Scope
    VAGUE_BROAD = "vague_and_overly_broad"
    UNMEASURABLE = "unmeasurable_terms"
    
    # Category 2: Spatial
    SPATIAL_HIERARCHY = "spatial_hierarchy_conflict"
    SPATIAL_OVERLAP = "spatial_overlap_conflict"
    
    # Category 3: Temporal
    TEMPORAL_EXPIRED = "temporal_expired_policy"
    TEMPORAL_OVERLAP = "temporal_overlap_conflict"
    TEMPORAL_IMPOSSIBLE = "temporal_impossible_sequence"
    
    
    # Category 4: Action Semantics
    ACTION_HIERARCHY = "action_hierarchy_conflict"
    ACTION_SUBSUMPTION = "action_subsumption_conflict"
    ACTION_CONFLICT = "action_conflict" 
    
    # Category 5: Dependencies
    CIRCULAR_DEPENDENCY = "circular_approval_dependency"
    WORKFLOW_CYCLE = "workflow_cycle_conflict"
    
    # Category 6: Roles & Context
    ROLE_HIERARCHY = "role_hierarchy_conflict"
    PARTY_INCONSISTENCY = "party_specification_inconsistency"

class ConflictDetectionStrategy(BaseModel):
    """Defines how to detect a specific conflict type"""
    conflict_type: ConflictType
    detection_order: int  # Lower = check first
    requires_ontology: bool = False
    requires_graph_analysis: bool = False
    
    # Detection patterns
    keyword_patterns: List[str] = []
    structural_patterns: List[Dict[str, Any]] = []
    
    # Resolution strategy
    resolution_principle: str  # "specific-over-general", "prohibit-on-ambiguity", etc.
    default_action: str  # "reject", "clarify", "warn"

# Detection strategies ordered by priority
CONFLICT_STRATEGIES = [
    # 1. VAGUE/UNMEASURABLE (check FIRST - highest priority)
    ConflictDetectionStrategy(
        conflict_type=ConflictType.UNMEASURABLE,
        detection_order=1,
        keyword_patterns=[
            "urgent", "soon", "later", "promptly", "quickly",
            "responsibly", "appropriately", "properly",
            "when necessary", "if important", "as needed",
            "everyone", "anyone", "nobody"
        ],
        resolution_principle="reject_with_measurable_alternative",
        default_action="reject"
    ),
    
    ConflictDetectionStrategy(
        conflict_type=ConflictType.VAGUE_BROAD,
        detection_order=2,
        keyword_patterns=[
            "everything", "anything", "all data", "any purpose",
            "everyone can access everything",
            "nobody can do anything"
        ],
        structural_patterns=[
            {"actors": "universal_quantifier", "assets": "universal_quantifier"},
            {"actors": "unspecified", "actions": "unspecified"}
        ],
        resolution_principle="require_specification",
        default_action="reject"
    ),
    
    # 2. TEMPORAL (check before spatial - affects all policies)
    ConflictDetectionStrategy(
        conflict_type=ConflictType.TEMPORAL_EXPIRED,
        detection_order=3,
        structural_patterns=[
            {"constraint_type": "temporal", "end_date": "before_current_date"}
        ],
        resolution_principle="flag_as_inactive",
        default_action="reject"
    ),
    
    ConflictDetectionStrategy(
        conflict_type=ConflictType.TEMPORAL_OVERLAP,
        detection_order=4,
        structural_patterns=[
            {"overlapping_intervals": True, "contradictory_actions": True}
        ],
        resolution_principle="specific_over_general",
        default_action="reject"
    ),
    
    # 3. SPATIAL (requires geographic ontology)
    ConflictDetectionStrategy(
        conflict_type=ConflictType.SPATIAL_HIERARCHY,
        detection_order=5,
        requires_ontology=True,
        structural_patterns=[
            {"narrow_scope": "permitted", "broad_scope": "prohibited", "containment": True}
        ],
        resolution_principle="specific_over_general",
        default_action="reject"
    ),
    
    # 4. ACTION SEMANTICS (requires ODRL action hierarchy)
    ConflictDetectionStrategy(
        conflict_type=ConflictType.ACTION_HIERARCHY,
        detection_order=6,
        requires_ontology=True,
        structural_patterns=[
            {"parent_action": "permitted", "child_action": "prohibited"},
            {"action_subsumption": True}
        ],
        resolution_principle="prohibit_on_conflict",
        default_action="reject"
    ),
    
    # 5. ROLE/PARTY (requires organizational hierarchy)
    ConflictDetectionStrategy(
        conflict_type=ConflictType.ROLE_HIERARCHY,
        detection_order=7,
        requires_ontology=True,
        structural_patterns=[
            {"broader_role": "required", "narrower_role": "prohibited", "role_containment": True}
        ],
        resolution_principle="apply_role_hierarchy",
        default_action="reject"
    ),
    
    # 6. DEPENDENCIES (requires graph analysis)
    ConflictDetectionStrategy(
        conflict_type=ConflictType.CIRCULAR_DEPENDENCY,
        detection_order=8,
        requires_graph_analysis=True,
        structural_patterns=[
            {"dependency_chain": "contains_cycle"}
        ],
        resolution_principle="break_cycle_at_weakest_link",
        default_action="reject"
    ),
]

# Resolution principles mapping
RESOLUTION_PRINCIPLES = {
    "specific_over_general": """
When two policies conflict, the more specific policy takes precedence:
- Narrower geographic scope > Broader scope
- Shorter time window > Longer window
- Specific actors > General groups
Example: "Germany only" overrides "All EU countries"
""",
    
    "prohibit_on_ambiguity": """
When temporal or spatial constraints create ambiguity, default to prohibition:
- Overlapping time windows with contradictory rules → Prohibit during overlap
- Conflicting geographic scopes → Prohibit in contested region
Safety principle: Better to block than allow incorrectly
""",
    
    "reject_with_measurable_alternative": """
Unmeasurable terms must be replaced with objective criteria:
- "urgent" → "priority level ≥ 5" or "within 48 hours of deadline"
- "responsibly" → "with proper attribution" or "for non-commercial purposes"
- "when necessary" → "when storage exceeds 80% capacity"
""",
    
    "require_specification": """
Overly broad policies must specify:
- Actors: Replace "everyone" with "registered researchers"
- Assets: Replace "everything" with specific dataset identifiers
- Actions: Replace "anything" with enumerated action list
""",
    
    "apply_role_hierarchy": """
Role conflicts resolved via organizational hierarchy:
1. Map role containment (managers ⊂ administrators)
2. Apply prohibition to all contained roles
3. Flag inconsistent specifications
""",
    
    "break_cycle_at_weakest_link": """
Circular dependencies broken by:
1. Detecting cycle via graph traversal
2. Identifying weakest dependency (least critical approval)
3. Suggesting alternative approval path
"""
}

class ConflictExample(BaseModel):
    """Concrete example of conflict for documentation"""
    user_input: str
    detected_conflict: ConflictType
    explanation: str
    resolution: str

# Examples for each conflict type
CONFLICT_EXAMPLES = {
    ConflictType.UNMEASURABLE: ConflictExample(
        user_input="Data must be used responsibly for urgent requests",
        detected_conflict=ConflictType.UNMEASURABLE,
        explanation="'Responsibly' and 'urgent' are subjective and cannot be measured consistently",
        resolution="Replace with: 'Data must be attributed to source and used only for requests submitted within 48 hours of deadline'"
    ),
    
    ConflictType.VAGUE_BROAD: ConflictExample(
        user_input="Everyone can access everything for any purpose",
        detected_conflict=ConflictType.VAGUE_BROAD,
        explanation="Universal quantifiers create unimplementable, overly permissive policy",
        resolution="Specify: 'Registered researchers can access datasets X, Y, Z for educational or non-commercial research purposes'"
    ),
    
    ConflictType.SPATIAL_HIERARCHY: ConflictExample(
        user_input="Access permitted in Germany but prohibited in all EU countries",
        detected_conflict=ConflictType.SPATIAL_HIERARCHY,
        explanation="Germany ⊂ EU, so permission and prohibition contradict",
        resolution="Apply specific-over-general: Allow in Germany (specific) despite EU prohibition (general), OR clarify intent"
    ),
    
    ConflictType.TEMPORAL_OVERLAP: ConflictExample(
        user_input="Access allowed 9am-5pm Monday to Friday, but prohibited 2pm-6pm every day",
        detected_conflict=ConflictType.TEMPORAL_OVERLAP,
        explanation="2pm-5pm overlap on weekdays has contradictory rules",
        resolution="Apply prohibit-on-ambiguity: Block access 2pm-5pm on weekdays"
    ),
    
    ConflictType.ACTION_HIERARCHY: ConflictExample(
        user_input="Users can share the dataset but cannot distribute it",
        detected_conflict=ConflictType.ACTION_HIERARCHY,
        explanation="In ODRL, 'share' is a subclass of 'distribute' (share ⊑ distribute)",
        resolution="Prohibit sharing since it's semantically a form of distribution"
    ),
    
    ConflictType.CIRCULAR_DEPENDENCY: ConflictExample(
        user_input="Access requires Committee approval → Committee needs Rights verification → Rights needs preliminary access",
        detected_conflict=ConflictType.CIRCULAR_DEPENDENCY,
        explanation="Cycle detected: Access → Committee → Rights → Access (impossible to start)",
        resolution="Break cycle: Allow preliminary access without Rights verification, OR delegate Rights verification to external party"
    ),
    
    ConflictType.ROLE_HIERARCHY: ConflictExample(
        user_input="Managers must access data weekly; administrators cannot access data; all managers are administrators",
        detected_conflict=ConflictType.ROLE_HIERARCHY,
        explanation="If managers ⊂ administrators, then requirement + prohibition is impossible",
        resolution="Clarify role hierarchy: Either managers ⊄ administrators, OR create exception for managers"
    ),
}

def get_detection_prompt_for_conflict_type(conflict_type: ConflictType) -> str:
    """Generate LLM prompt section for specific conflict type"""
    strategy = next(s for s in CONFLICT_STRATEGIES if s.conflict_type == conflict_type)
    example = CONFLICT_EXAMPLES.get(conflict_type)
    principle = RESOLUTION_PRINCIPLES.get(strategy.resolution_principle, "")
    
    prompt = f"""
## {conflict_type.value.replace('_', ' ').title()}

**Detection Order:** {strategy.detection_order} (Priority: {'CRITICAL' if strategy.detection_order <= 2 else 'High' if strategy.detection_order <= 5 else 'Standard'})

**Keywords to Check:** {', '.join(strategy.keyword_patterns) if strategy.keyword_patterns else 'N/A - structural patterns only'}

**Resolution Principle:** {strategy.resolution_principle}
{principle}

**Example:**
Input: "{example.user_input if example else 'N/A'}"
Conflict: {example.explanation if example else 'N/A'}
Resolution: {example.resolution if example else 'N/A'}
"""
    return prompt