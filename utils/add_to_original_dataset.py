"""
Add new spatial policy to original dataset format
"""

import json
from pathlib import Path

NEW_POLICY_ORIGINAL_FORMAT = {
    "policy_id": "conflict_spatial_004",
    "policy_text": "Access to the 'ArchivalRecords' dataset is granted to researchers located in Bavaria. However, all access from German federal states is strictly prohibited for data protection compliance.",
    "expected_outcome": "REJECTED",
    "rejection_category": "spatial_hierarchy",
    "rejection_category_description": "Policy creates geographical contradiction through regional containment",
    "specific_contradiction": "Bavaria is a federal state within Germany (Bavaria ⊂ Germany), so Bavarian researchers are both permitted (as Bavaria residents) and prohibited (as German federal state residents)",
    "recommendation": "Either exempt Bavaria from the German federal states prohibition, or remove the Bavaria-specific permission. Clarify geographical scope boundaries.",
    "rejection_reason_detailed": "This policy creates a spatial hierarchy contradiction similar to the Germany-EU case but at the national-regional level. Bavaria (Bayern) is one of the 16 federal states (Bundesländer) of Germany. Granting access to Bavarian researchers while prohibiting all access from German federal states creates a direct geographical contradiction where the same set of researchers is simultaneously granted permission (by virtue of being in Bavaria) and denied permission (by virtue of Bavaria being a German federal state). The policy is impossible to enforce as it creates overlapping and contradictory spatial constraints."
}

def add_to_original():
    """Add to original dataset"""
    
    input_file = Path("data/rejected_policies/rejected_policies_dataset.json")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        policies = json.load(f)
    
    # Check if already exists
    existing_ids = {p.get("policy_id", "") for p in policies}
    if NEW_POLICY_ORIGINAL_FORMAT["policy_id"] in existing_ids:
        print(f"Policy already exists!")
        return
    
    # Add new policy
    policies.append(NEW_POLICY_ORIGINAL_FORMAT)
    
    # Save
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(policies, f, indent=2, ensure_ascii=False)
    
    print(f" Added policy to original dataset")
    print(f" Total policies: {len(policies)}")
    print(f" Saved to: {input_file}")
    print("\nRemember to run unify_rejected_policies.py to update unified format!")


if __name__ == "__main__":
    add_to_original()