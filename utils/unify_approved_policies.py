# scripts/unify_approved_policies.py
"""
Unify all approved policies into standardized evaluation format
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def convert_approved_policy(policy: dict) -> dict:
    """Convert approved policy from old format to unified format"""
    
    policy_id = policy.get("policy_id", "unknown")
    policy_text = policy.get("policy_text", "")
    acceptance_category = policy.get("acceptance_category", "general_policy")
    
    # Extract source from policy_id
    source = "synthetic"
    if "drk" in policy_id.lower():
        source = "DRK"
    elif "ids" in policy_id.lower():
        source = "IDS"
    elif "mds" in policy_id.lower() or "mobility" in policy_id.lower():
        source = "MDS"
    elif "cc_" in policy_id.lower():
        source = "CC"  # Creative Commons examples
    
    # Determine category based on acceptance_category
    category_mapping = {
        "technical_constraint_policy": "connector_restrictions",
        "role_constraint_policy": "role_based_access",
        "temporal_constraint_policy": "temporal_constraints",
        "purpose_constraint_policy": "purpose_restrictions",
        "financial_obligation_policy": "financial_obligations",
        "conditional_access_policy": "conditional_access",
        "semantic_class_constraint_policy": "semantic_constraints",
        "multi_constraint_policy": "multi_constraint",
        "quantitative_limit_policy": "usage_limits",
        "exclusive_access_policy": "exclusive_access",
        "usage_count_policy": "count_constraints",
        "data_transformation_policy": "data_transformation",
        "multi_dimension_access_policy": "multi_dimension",
        "public_access_policy": "public_access",
        "purpose_based_fee_waiver_policy": "fee_waiver",
        "authentication_requirement_policy": "authentication",
        "complex_multi_stage_policy": "multi_stage_workflow",
        "activity_logging_policy": "activity_logging",
        "deletion_requirement_policy": "deletion_requirements",
        "up_to_dateness_policy": "data_freshness",
        "data_quality_policy": "data_quality",
        "data_aggregation_policy": "data_aggregation",
        "bandwidth_constraint_policy": "bandwidth_limits",
        "concurrent_connection_constraint_policy": "connection_limits",
        "membership_constraint_policy": "membership_restrictions",
        "multi_stage_access_policy": "multi_stage_access",
        "creative_commons_policy": "creative_commons_licensing",
        "duration_constraint_policy": "duration_constraints",
        "temporal_and_post_duty_policy": "temporal_with_post_duty",
        "logging_policy": "usage_logging",
        "notification_policy": "usage_notification",
        "connector_restriction_policy": "connector_restrictions",
        "security_profile_policy": "security_requirements",
        "location_restriction_policy": "location_restrictions",
        "event_restriction_policy": "event_restrictions",
        "payment_based_policy": "payment_requirements",
        "provide_access": "basic_access",
        "connector_restricted_usage": "connector_restrictions",
        "application_restricted_usage": "application_restrictions",
        "interval_restricted_usage": "temporal_constraints",
        "duration_restricted_usage": "duration_constraints",
        "location_restricted_usage": "location_restrictions",
        "perpetual_data_sale": "perpetual_access",
        "data_rental": "rental_access",
        "role_restricted_usage": "role_based_access",
        "purpose_restricted_usage": "purpose_restrictions",
        "event_restricted_usage": "event_restrictions",
        "restricted_number_of_usages": "count_constraints",
        "security_level_restricted_usage": "security_requirements",
        "use_data_and_delete_after": "temporal_with_deletion",
        "modify_data_in_transit": "data_modification_transit",
        "modify_data_in_rest": "data_modification_rest",
        "local_logging": "activity_logging",
        "remote_notifications": "remote_notification",
        "prohibit_access": "access_prohibition"
    }
    
    category = category_mapping.get(acceptance_category, acceptance_category)
    
    # Detect ODRL features from policy text (simple heuristics)
    odrl_features = []
    text_lower = policy_text.lower()
    
    if any(word in text_lower for word in ["until", "between", "before", "after", "expires"]):
        odrl_features.append("temporal_constraint")
    if any(word in text_lower for word in ["maximum", "minimum", "up to", "at most", "at least"]):
        odrl_features.append("count_constraint")
    if any(word in text_lower for word in ["purpose", "educational", "research", "commercial", "non-commercial"]):
        odrl_features.append("purpose_constraint")
    if any(word in text_lower for word in ["role", "researcher", "curator", "member"]):
        odrl_features.append("role_constraint")
    if any(word in text_lower for word in ["location", "germany", "within", "region"]):
        odrl_features.append("spatial_constraint")
    if any(word in text_lower for word in ["connector", "via", "through"]):
        odrl_features.append("connector_constraint")
    if any(word in text_lower for word in ["log", "notify", "inform", "report"]):
        odrl_features.append("duty_logging")
    if any(word in text_lower for word in ["delete", "remove", "destroy"]):
        odrl_features.append("duty_deletion")
    if any(word in text_lower for word in ["fee", "payment", "pay", "compensate", "euros"]):
        odrl_features.append("financial_constraint")
    
    # Determine permission type
    if "prohibit" in text_lower or "denied" in text_lower or "not allowed" in text_lower:
        permission_type = "prohibition"
    else:
        permission_type = "permission"
    
    # Detect complexity
    constraint_count = len(odrl_features)
    if constraint_count <= 1:
        complexity = "simple"
    elif constraint_count <= 3:
        complexity = "moderate"
    else:
        complexity = "complex"
    
    return {
        "policy_id": policy_id,
        "policy_text": policy_text,
        "source": source,
        "category": category,
        
        "ground_truth": {
            "expected_outcome": "APPROVED",
            "conflicts": [],
            "conflict_primary": None
        },
        
        "metadata": {
            "acceptance_explanation": policy.get("acceptance_reasoning_detailed", ""),
            "acceptance_reason": policy.get("acceptance_reason", ""),
            "original_category": acceptance_category,
            "odrl_features": odrl_features,
            "permission_type": permission_type,
            "complexity": complexity,
            "expected_shacl": "valid"
        }
    }


def main():
    """Main conversion function"""
    
    print("=" * 80)
    print("UNIFYING APPROVED POLICIES DATASET")
    print("=" * 80)
    
    # Load existing approved policies
    input_file = Path("data/approved_policies/approved_policies_dataset.json")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        old_policies = json.load(f)
    
    print(f"\n📥 Loaded {len(old_policies)} policies")
    
    # Convert all policies
    unified_policies = []
    for policy in old_policies:
        try:
            converted = convert_approved_policy(policy)
            unified_policies.append(converted)
        except Exception as e:
            print(f"⚠️  Warning: Failed to convert {policy.get('policy_id', 'unknown')}: {e}")
    
    # Calculate statistics
    category_distribution = defaultdict(int)
    source_distribution = defaultdict(int)
    complexity_distribution = defaultdict(int)
    feature_distribution = defaultdict(int)
    
    for policy in unified_policies:
        category_distribution[policy["category"]] += 1
        source_distribution[policy["source"]] += 1
        complexity_distribution[policy["metadata"]["complexity"]] += 1
        
        for feature in policy["metadata"]["odrl_features"]:
            feature_distribution[feature] += 1
    
    # Create output structure
    output = {
        "dataset_info": {
            "name": "ODRL Valid Policy Dataset - Unified Format",
            "version": "1.0",
            "total_policies": len(unified_policies),
            "creation_date": datetime.now().strftime("%Y-%m-%d"),
            "description": "Standardized approved policies for generation and validation evaluation",
            
            "source_distribution": dict(source_distribution),
            
            "category_summary": {
                "total_categories": len(category_distribution),
                "top_categories": dict(sorted(category_distribution.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            
            "complexity_distribution": dict(complexity_distribution),
            
            "feature_distribution": dict(sorted(feature_distribution.items(), key=lambda x: x[1], reverse=True))
        },
        
        "policies": unified_policies
    }
    
    # Save unified format
    output_file = Path("data/approved_policies/approved_policies_unified.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\nSuccessfully converted {len(unified_policies)} policies")
    print(f"Saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("SOURCE DISTRIBUTION")
    print("=" * 80)
    for source, count in sorted(source_distribution.items()):
        percentage = (count / len(unified_policies)) * 100
        print(f"{source:20s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("COMPLEXITY DISTRIBUTION")
    print("=" * 80)
    for complexity, count in sorted(complexity_distribution.items()):
        percentage = (count / len(unified_policies)) * 100
        print(f"{complexity:20s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("TOP 10 POLICY CATEGORIES")
    print("=" * 80)
    for category, count in sorted(category_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(unified_policies)) * 100
        print(f"{category:40s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("ODRL FEATURES DISTRIBUTION")
    print("=" * 80)
    for feature, count in sorted(feature_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feature:30s}: {count:3d}")
    
    print("\nUnification complete!")


if __name__ == "__main__":
    main()