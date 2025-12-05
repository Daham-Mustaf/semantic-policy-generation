# scripts/unify_rejected_policies.py
"""
Unify all rejected policies into standardized evaluation format
Maps all conflict types to the 6 main categories used in the reasoning agent
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ===== CONFLICT TYPE MAPPING =====
# Maps specific rejection categories to the 6 main conflict types in your implementation

CONFLICT_TYPE_MAPPING = {
    # VAGUENESS conflicts
    "internal_purpose_contradiction": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "exclusive_purpose_contradiction"
    },
    "overly_broad_policy": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "universal_quantifier_without_specificity"
    },
    "conditional_ambiguity": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "undefined_condition_criteria"
    },
    "universal_quantifier_contradiction": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "universal_existential_conflict"
    },
    "existential_universal_quantifier_conflict": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "quantifier_scope_contradiction"
    },
    "universal_existential_ambiguity": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "undefined_existential_restriction"
    },
    "universal_existential_action_conflict": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "universal_deletion_existential_preservation"
    },
    "nested_universal_quantifier_contradiction": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "nested_quantifier_subset_conflict"
    },
    "negated_constraint_contradiction": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "logical_negation_conflict"
    },
    
    # TEMPORAL conflicts
    "temporal_conflict": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "contradictory_time_windows"
    },
    "temporal_access_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "overlapping_access_windows"
    },
    "expired_time_condition": {
        "primary": "temporal",
        "specific": ["temporal_expired_policy"],
        "pattern": "past_deadline_policy"
    },
    "time_constraint_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "definite_indefinite_time_conflict"
    },
    "temporal_duty_precedence_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_impossible_sequence"],
        "pattern": "duty_before_permission_activation"
    },
    "temporal_elapsed_time_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "deletion_before_permission_expiry"
    },
    "temporal_duty_verification_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_impossible_sequence"],
        "pattern": "verification_after_deadline"
    },
    "temporal_consequence_precedence_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_impossible_sequence"],
        "pattern": "consequence_before_trigger"
    },
    "temporal_circular_dependency": {
        "primary": "circular_dependency",
        "specific": ["circular_approval_dependency"],
        "pattern": "temporal_causality_loop"
    },
    "temporal_constraint_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "required_date_outside_permitted_range"
    },
    
    # SPATIAL conflicts
    "hierarchical_location_contradiction": {
        "primary": "spatial",
        "specific": ["spatial_hierarchy_conflict"],
        "pattern": "geographic_containment_violation"
    },
    "geographic_contradiction": {
        "primary": "spatial",
        "specific": ["spatial_hierarchy_conflict"],
        "pattern": "subset_region_contradiction"
    },
    
    # ACTION HIERARCHY conflicts
    "action_hierarchy_contradiction": {
        "primary": "action_hierarchy",
        "specific": ["action_hierarchy_conflict"],
        "pattern": "odrl_action_subsumption_violation"
    },
    "action_permission_contradiction": {
        "primary": "action_hierarchy",
        "specific": ["action_hierarchy_conflict"],
        "pattern": "direct_action_contradiction"
    },
    "action_refinement_contradiction": {
        "primary": "action_hierarchy",
        "specific": ["action_hierarchy_conflict"],
        "pattern": "action_quality_conflict"
    },
    "action_refinement_specificity_conflict": {
        "primary": "action_hierarchy",
        "specific": ["action_hierarchy_conflict"],
        "pattern": "general_specific_refinement_conflict"
    },
    "action_refinement_range_contradiction": {
        "primary": "action_hierarchy",
        "specific": ["action_hierarchy_conflict"],
        "pattern": "quality_range_non_overlap"
    },
    
    # ROLE/PARTY conflicts
    "role_hierarchy_contradiction": {
        "primary": "role_hierarchy",
        "specific": ["role_hierarchy_conflict"],
        "pattern": "role_containment_violation"
    },
    "actor_permission_contradiction": {
        "primary": "role_hierarchy",
        "specific": ["party_specification_inconsistency"],
        "pattern": "exclusive_inclusive_actor_conflict"
    },
    "party_type_subsumption_contradiction": {
        "primary": "role_hierarchy",
        "specific": ["role_hierarchy_conflict"],
        "pattern": "party_type_hierarchy_violation"
    },
    "role_classification_contradiction": {
        "primary": "role_hierarchy",
        "specific": ["role_hierarchy_conflict"],
        "pattern": "specialized_role_parent_conflict"
    },
    "role_duty_prohibition_contradiction": {
        "primary": "role_hierarchy",
        "specific": ["role_hierarchy_conflict"],
        "pattern": "duty_prohibition_hierarchy_conflict"
    },
    "party_relationship_inclusion_contradiction": {
        "primary": "role_hierarchy",
        "specific": ["party_specification_inconsistency"],
        "pattern": "group_subgroup_access_conflict"
    },
    
    # CIRCULAR DEPENDENCY conflicts
    "circular_approval_dependency": {
        "primary": "circular_dependency",
        "specific": ["circular_approval_dependency"],
        "pattern": "circular_prerequisite_chain"
    },
    "circular_dependency_contradiction": {
        "primary": "circular_dependency",
        "specific": ["circular_approval_dependency"],
        "pattern": "cross_dataset_circular_dependency"
    },
    "contradictory_duty_state_requirements": {
        "primary": "circular_dependency",
        "specific": ["circular_approval_dependency"],
        "pattern": "mutually_exclusive_duty_states"
    },
    
    # OTHER (mapped to closest category)
    "usage_limit_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "limited_unlimited_usage_conflict"
    },
    "legal_compliance_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "retention_deletion_conflict"
    },
    "technical_feasibility_contradiction": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "technically_impossible_requirement"
    },
    "incomplete_condition_handling": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "incomplete_condition_coverage"
    },
    "resource_constraint_contradiction": {
        "primary": "vagueness",
        "specific": ["unmeasurable_terms"],
        "pattern": "infeasible_resource_requirements"
    },
    "numeric_constraint_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "non_overlapping_numeric_range"
    },
    "payment_constraint_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "free_paid_contradiction"
    },
    "constraint_range_overlap_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "permission_prohibition_range_overlap"
    },
    "set_operator_contradiction": {
        "primary": "spatial",
        "specific": ["spatial_hierarchy_conflict"],
        "pattern": "set_membership_contradiction"
    },
    "percentage_constraint_contradiction": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "percentage_usage_conflict"
    },
    "asset_refinement_overlap_contradiction": {
        "primary": "action_hierarchy",
        "specific": ["action_hierarchy_conflict"],
        "pattern": "asset_scope_overlap"
    },
    "asset_refinement_scope_mismatch": {
        "primary": "action_hierarchy",
        "specific": ["action_hierarchy_conflict"],
        "pattern": "disjoint_refinement_scopes"
    },
    "activity_logging_policy": {
        "primary": "action_hierarchy",
        "specific": ["action_hierarchy_conflict"],
        "pattern": "inform_prohibit_inform_conflict"
    },
    "deletion_requirement_policy": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "delete_before_retain_until"
    },
    "data_quality_policy": {
        "primary": "action_hierarchy",
        "specific": ["action_hierarchy_conflict"],
        "pattern": "conform_not_conform_shacl"
    },
    "data_aggregation_policy": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "percentage_aggregation_conflict"
    },
    "bandwidth_constraint_policy": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "bandwidth_limit_conflict"
    },
    "concurrent_connection_constraint_policy": {
        "primary": "temporal",
        "specific": ["temporal_overlap"],
        "pattern": "connection_limit_conflict"
    }
}


def convert_policy_to_unified(policy: dict) -> dict:
    """Convert a policy from old format to unified format"""
    
    policy_id = policy.get("policy_id", "unknown")
    policy_text = policy.get("policy_text", "")
    rejection_category = policy.get("rejection_category", policy.get("acceptance_category", "unknown"))
    
    # Map to conflict type
    conflict_mapping = CONFLICT_TYPE_MAPPING.get(
        rejection_category,
        {
            "primary": "vagueness",
            "specific": ["unmeasurable_terms"],
            "pattern": "uncategorized_conflict"
        }
    )
    
    # Extract source from policy_id
    source = "synthetic"
    if "drk" in policy_id.lower():
        source = "DRK"
    elif "ids" in policy_id.lower():
        source = "IDS"
    elif "mds" in policy_id.lower():
        source = "MDS"
    
    return {
        "policy_id": policy_id,
        "policy_text": policy_text,
        "source": source,
        "category": rejection_category,
        
        "ground_truth": {
            "expected_outcome": policy.get("expected_outcome", "REJECTED"),
            "conflicts": conflict_mapping["specific"],
            "conflict_primary": conflict_mapping["primary"]
        },
        
        "metadata": {
            "contradiction_explanation": policy.get(
                "rejection_reason_detailed",
                policy.get("specific_contradiction", "")
            ),
            "recommendation": policy.get("recommendation", ""),
            "conflict_pattern": conflict_mapping["pattern"],
            "original_category": rejection_category
        }
    }


def main():
    """Main conversion function"""
    
    print("=" * 80)
    print("UNIFYING REJECTED POLICIES DATASET")
    print("=" * 80)
    
    # Load existing rejected policies
    input_file = Path("data/rejected_policies/rejected_policies_dataset.json")
    
    if not input_file.exists():
        print(f" Error: {input_file} not found!")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        old_policies = json.load(f)
    
    print(f"\n📥 Loaded {len(old_policies)} policies")
    
    # Convert all policies
    unified_policies = []
    for policy in old_policies:
        try:
            converted = convert_policy_to_unified(policy)
            unified_policies.append(converted)
        except Exception as e:
            print(f"⚠️  Warning: Failed to convert {policy.get('policy_id', 'unknown')}: {e}")
    
    # Calculate statistics
    distribution = defaultdict(int)
    source_distribution = defaultdict(int)
    pattern_distribution = defaultdict(int)
    
    for policy in unified_policies:
        distribution[policy["ground_truth"]["conflict_primary"]] += 1
        source_distribution[policy["source"]] += 1
        pattern_distribution[policy["metadata"]["conflict_pattern"]] += 1
    
    # Create output structure
    output = {
        "dataset_info": {
            "name": "ODRL Conflict Detection Dataset - Unified Format",
            "version": "1.0",
            "total_policies": len(unified_policies),
            "creation_date": datetime.now().strftime("%Y-%m-%d"),
            "description": "Standardized rejected policies for conflict detection evaluation",
            
            "conflict_distribution": {
                "vagueness": distribution["vagueness"],
                "temporal": distribution["temporal"],
                "spatial": distribution["spatial"],
                "action_hierarchy": distribution["action_hierarchy"],
                "role_hierarchy": distribution["role_hierarchy"],
                "circular_dependency": distribution["circular_dependency"]
            },
            
            "source_distribution": dict(source_distribution),
            
            "conflict_type_mapping": {
                "vagueness": ["unmeasurable_terms"],
                "temporal": ["temporal_overlap", "temporal_expired_policy", "temporal_impossible_sequence"],
                "spatial": ["spatial_hierarchy_conflict", "spatial_overlap_conflict"],
                "action_hierarchy": ["action_hierarchy_conflict", "action_subsumption_conflict"],
                "role_hierarchy": ["role_hierarchy_conflict", "party_specification_inconsistency"],
                "circular_dependency": ["circular_approval_dependency", "workflow_cycle_conflict"]
            }
        },
        
        "policies": unified_policies
    }
    
    # Save unified format
    output_file = Path("data/rejected_policies/rejected_policies_unified.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\n✅ Successfully converted {len(unified_policies)} policies")
    print(f"💾 Saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("CONFLICT DISTRIBUTION")
    print("=" * 80)
    for conflict_type, count in sorted(distribution.items()):
        percentage = (count / len(unified_policies)) * 100
        print(f"{conflict_type:20s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("SOURCE DISTRIBUTION")
    print("=" * 80)
    for source, count in sorted(source_distribution.items()):
        percentage = (count / len(unified_policies)) * 100
        print(f"{source:20s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("TOP 10 CONFLICT PATTERNS")
    print("=" * 80)
    for pattern, count in sorted(pattern_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{pattern:50s}: {count:3d}")
    
    print("\n✅ Unification complete!")


if __name__ == "__main__":
    main()