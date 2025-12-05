
"""
Add one more spatial conflict policy to balance the dataset
"""

import json
from pathlib import Path
from datetime import datetime

# New spatial conflict policy
NEW_SPATIAL_POLICY = {
    "policy_id": "conflict_spatial_004",
    "policy_text": "Access to the 'ArchivalRecords' dataset is granted to researchers located in Bavaria. However, all access from German federal states is strictly prohibited for data protection compliance.",
    "source": "DRK",
    "category": "spatial_hierarchy",
    
    "ground_truth": {
        "expected_outcome": "REJECTED",
        "conflicts": ["spatial_hierarchy_conflict"],
        "conflict_primary": "spatial"
    },
    
    "metadata": {
        "contradiction_explanation": "Bavaria is a federal state of Germany (Bavaria ⊂ Germany). Granting access to Bavarian researchers while prohibiting access from German federal states creates a geographical contradiction where Bavarian researchers are simultaneously permitted and prohibited.",
        "recommendation": "Either exempt Bavaria from the German federal states prohibition or remove the Bavaria-specific permission. Clarify whether the prohibition applies to all German states or only specific ones.",
        "conflict_pattern": "regional_containment_contradiction",
        "original_category": "spatial_hierarchy"
    }
}

def add_policy_to_dataset():
    """Add new policy to unified dataset"""
    
    # Load existing unified dataset
    input_file = Path("data/rejected_policies/rejected_policies_unified.json")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print("   Please run unify_rejected_policies.py first")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print("=" * 80)
    print("ADDING NEW SPATIAL CONFLICT POLICY")
    print("=" * 80)
    
    # Check if policy already exists
    existing_ids = {p["policy_id"] for p in dataset["policies"]}
    if NEW_SPATIAL_POLICY["policy_id"] in existing_ids:
        print(f" Policy {NEW_SPATIAL_POLICY['policy_id']} already exists!")
        return
    
    # Add new policy
    dataset["policies"].append(NEW_SPATIAL_POLICY)
    
    # Update statistics
    dataset["dataset_info"]["total_policies"] += 1
    dataset["dataset_info"]["conflict_distribution"]["spatial"] += 1
    dataset["dataset_info"]["source_distribution"]["DRK"] += 1
    dataset["dataset_info"]["creation_date"] = datetime.now().strftime("%Y-%m-%d")
    
    # Save updated dataset
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nAdded policy: {NEW_SPATIAL_POLICY['policy_id']}")
    print(f"Policy text: {NEW_SPATIAL_POLICY['policy_text'][:80]}...")
    print(f"\nUpdated dataset saved to: {input_file}")
    
    print("\n" + "=" * 80)
    print("UPDATED CONFLICT DISTRIBUTION")
    print("=" * 80)
    for conflict_type, count in sorted(dataset["dataset_info"]["conflict_distribution"].items()):
        total = dataset["dataset_info"]["total_policies"]
        percentage = (count / total) * 100
        print(f"{conflict_type:20s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("UPDATED SOURCE DISTRIBUTION")
    print("=" * 80)
    for source, count in sorted(dataset["dataset_info"]["source_distribution"].items()):
        total = dataset["dataset_info"]["total_policies"]
        percentage = (count / total) * 100
        print(f"{source:20s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\nDataset update complete!")
    print(f"Total policies: {dataset['dataset_info']['total_policies']}")


if __name__ == "__main__":
    add_policy_to_dataset()