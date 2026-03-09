# evaluation/evaluate_reasoning_agent.py

"""
Evaluate YOUR Reasoning Agent on the unified dataset
Tests the 6-phase structured conflict detection
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from agents.reasoner.reasoner_agent import Reasoner
from evaluation.model_config_loader import load_model_config


@dataclass
class SimpleResult:
    """Simple result"""
    policy_id: str
    expected: str  # raw expected outcome string from dataset
    expected_binary: Optional[str]  # "APPROVED" | "REJECTED" | None (ambiguous)
    agent_decision: str  # "approve", "reject", "needs_input"
    correct: bool
    expected_conflicts: List[str]
    predicted_conflicts: List[str]
    primary_conflict_match: Optional[bool]


@dataclass
class SimpleMetrics:
    """Simple metrics"""
    total: int
    total_binary_evaluated: int
    skipped_ambiguous: int
    correct: int
    accuracy: float
    
    should_reject: int
    correctly_rejected: int
    missed: int
    rejection_accuracy: float
    
    should_approve: int
    correctly_approved: int
    over_rejected: int
    approval_accuracy: float
    conflict_distribution: Dict[str, int]
    conflict_type_samples: int
    conflict_type_primary_correct: int
    conflict_type_primary_accuracy: float
    conflict_type_precision: float
    conflict_type_recall: float
    conflict_type_f1: float
    missing_conflict_predictions: int


CONFLICT_TYPE_ALIASES = {
    "temporal_overlap": "temporal_overlap_conflict",
    "temporal_conflict": "temporal_overlap_conflict",
    "expired_policy": "temporal_expired_policy",
    "temporal_expired": "temporal_expired_policy",
    "spatial_conflict": "spatial_hierarchy_conflict",
    "action_conflict_generic": "action_conflict",
    "actor_conflict": "party_specification_inconsistency",
    "cross_policy_conflict": "party_specification_inconsistency",
    "usage_limit_conflict": "temporal_overlap_conflict",
    "role_conflict": "role_hierarchy_conflict",
    "circular_dependency": "circular_approval_dependency",
    "vague_term": "unmeasurable_terms",
    "vague_terms": "unmeasurable_terms",
    "conflict": "action_conflict",
    "constraint_conflict": "action_conflict",
    "overly_broad_policy": "vague_and_overly_broad",
    "overly_broad": "vague_and_overly_broad",
    "unenforceable": "unmeasurable_terms",
    "technical_impossibility": "action_conflict",
    "workflow_cycle": "workflow_cycle_conflict",
}


def normalize_conflict_type(value: Optional[str]) -> Optional[str]:
    """Normalize conflict labels to canonical taxonomy strings."""
    if not value:
        return None
    normalized = str(value).strip().lower()
    return CONFLICT_TYPE_ALIASES.get(normalized, normalized)


def normalize_expected_decision(raw_expected: str) -> Optional[str]:
    """
    Normalize expected decision label.
    Returns None when label is ambiguous (contains both approve/reject).
    """
    text = str(raw_expected or "").strip().upper()
    has_approve = "APPROVED" in text
    has_reject = "REJECTED" in text
    if has_approve and has_reject:
        return None
    if has_approve:
        return "APPROVED"
    if has_reject:
        return "REJECTED"
    return None


def evaluate_policy(reasoner: Reasoner, policy: dict) -> SimpleResult:
    """Evaluate one policy"""
    
    policy_id = policy["policy_id"]
    policy_text = policy["policy_text"]
    expected = policy["ground_truth"]["expected_outcome"]
    expected_binary = normalize_expected_decision(expected)
    expected_conflicts = sorted({
        c for c in (
            normalize_conflict_type(c)
            for c in policy.get("ground_truth", {}).get("conflicts", [])
        ) if c
    })
    
    try:
        # Call agent
        result = reasoner.reason(policy_text)
        agent_decision = result.get("decision", "needs_input").lower()
        issues = result.get("issues", [])
        predicted_conflicts = sorted({
            c for c in (
                normalize_conflict_type(issue.get("conflict_type") or issue.get("category"))
                for issue in issues
            ) if c
        })

        primary_conflict_match = None
        if expected_binary == "REJECTED" and expected_conflicts:
            predicted_primary = predicted_conflicts[0] if predicted_conflicts else None
            primary_conflict_match = predicted_primary in set(expected_conflicts) if predicted_primary else False

        # Check binary decision correctness
        if expected_binary == "APPROVED":
            correct = (agent_decision == "approve")
        elif expected_binary == "REJECTED":
            correct = (agent_decision == "reject")
        else:
            # Skip ambiguous labels in binary decision accuracy.
            correct = False

        return SimpleResult(
            policy_id=policy_id,
            expected=expected,
            expected_binary=expected_binary,
            agent_decision=agent_decision,
            correct=correct,
            expected_conflicts=expected_conflicts,
            predicted_conflicts=predicted_conflicts,
            primary_conflict_match=primary_conflict_match
        )
        
    except Exception as e:
        print(f"\n Error: {policy_id}: {str(e)[:100]}")
        return SimpleResult(
            policy_id=policy_id,
            expected=expected,
            expected_binary=expected_binary,
            agent_decision="ERROR",
            correct=False,
            expected_conflicts=expected_conflicts,
            predicted_conflicts=[],
            primary_conflict_match=False if expected_binary == "REJECTED" and expected_conflicts else None
        )


def calculate_metrics(results: List[SimpleResult]) -> SimpleMetrics:
    """Calculate metrics"""
    
    total = len(results)
    binary_results = [r for r in results if r.expected_binary in {"APPROVED", "REJECTED"}]
    total_binary_evaluated = len(binary_results)
    skipped_ambiguous = total - total_binary_evaluated

    correct = sum(1 for r in binary_results if r.correct)
    
    should_reject = [r for r in binary_results if r.expected_binary == "REJECTED"]
    should_approve = [r for r in binary_results if r.expected_binary == "APPROVED"]
    
    correctly_rejected = sum(1 for r in should_reject if r.agent_decision == "reject")
    missed = len(should_reject) - correctly_rejected
    
    correctly_approved = sum(1 for r in should_approve if r.agent_decision == "approve")
    over_rejected = sum(1 for r in should_approve if r.agent_decision == "reject")

    conflict_counter = Counter()
    for r in results:
        conflict_counter.update(r.predicted_conflicts)

    # Conflict-type metrics on rejected samples with gold conflict labels.
    conflict_eval_samples = [
        r for r in results
        if r.expected_binary == "REJECTED" and len(r.expected_conflicts) > 0
    ]
    conflict_type_samples = len(conflict_eval_samples)
    conflict_type_primary_correct = sum(1 for r in conflict_eval_samples if r.primary_conflict_match)
    missing_conflict_predictions = sum(1 for r in conflict_eval_samples if len(r.predicted_conflicts) == 0)

    tp = fp = fn = 0
    for r in conflict_eval_samples:
        gold = set(r.expected_conflicts)
        pred = set(r.predicted_conflicts)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    
    return SimpleMetrics(
        total=total,
        total_binary_evaluated=total_binary_evaluated,
        skipped_ambiguous=skipped_ambiguous,
        correct=correct,
        accuracy=(correct / total_binary_evaluated * 100) if total_binary_evaluated > 0 else 0,
        should_reject=len(should_reject),
        correctly_rejected=correctly_rejected,
        missed=missed,
        rejection_accuracy=(correctly_rejected / len(should_reject) * 100) if should_reject else 0,
        should_approve=len(should_approve),
        correctly_approved=correctly_approved,
        over_rejected=over_rejected,
        approval_accuracy=(correctly_approved / len(should_approve) * 100) if should_approve else 0,
        conflict_distribution=dict(conflict_counter),
        conflict_type_samples=conflict_type_samples,
        conflict_type_primary_correct=conflict_type_primary_correct,
        conflict_type_primary_accuracy=(conflict_type_primary_correct / conflict_type_samples * 100) if conflict_type_samples else 0,
        conflict_type_precision=precision * 100,
        conflict_type_recall=recall * 100,
        conflict_type_f1=f1 * 100,
        missing_conflict_predictions=missing_conflict_predictions
    )


def print_results(metrics: SimpleMetrics):
    """Print results"""
    
    print("\n" + "="*80)
    print("REASONING AGENT RESULTS")
    print("="*80)
    print(f"Total:             {metrics.total}")
    print(f"Binary evaluated:  {metrics.total_binary_evaluated}")
    print(f"Skipped ambiguous: {metrics.skipped_ambiguous}")
    print(f"Correct:           {metrics.correct}")
    print(f"Accuracy:          {metrics.accuracy:.1f}%")
    
    print("\n" + "="*80)
    print("REJECTION TEST (Catching Bad Policies)")
    print("="*80)
    print(f"Should Reject:     {metrics.should_reject}")
    print(f"Rejected:        {metrics.correctly_rejected}")
    print(f"Missed:          {metrics.missed}")
    print(f"Accuracy:          {metrics.rejection_accuracy:.1f}%")
    
    print("\n" + "="*80)
    print("APPROVAL TEST (Not Over-rejecting)")
    print("="*80)
    print(f"Should Approve:    {metrics.should_approve}")
    print(f"Approved:        {metrics.correctly_approved}")
    print(f"Over-rejected:   {metrics.over_rejected}")
    print(f"Accuracy:          {metrics.approval_accuracy:.1f}%")

    print("\n" + "="*80)
    print("CONFLICT TYPE EVALUATION (REJECTED ONLY)")
    print("="*80)
    print(f"Samples:           {metrics.conflict_type_samples}")
    print(f"Primary match:     {metrics.conflict_type_primary_correct}")
    print(f"Primary accuracy:  {metrics.conflict_type_primary_accuracy:.1f}%")
    print(f"Type precision:    {metrics.conflict_type_precision:.1f}%")
    print(f"Type recall:       {metrics.conflict_type_recall:.1f}%")
    print(f"Type F1:           {metrics.conflict_type_f1:.1f}%")
    print(f"No type predicted: {metrics.missing_conflict_predictions}")

    if metrics.conflict_distribution:
        print("\n" + "="*80)
        print("DETECTED CONFLICT TYPE DISTRIBUTION")
        print("="*80)
        total = sum(metrics.conflict_distribution.values())
        for conflict_type, count in sorted(
            metrics.conflict_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = (count / total * 100) if total else 0
            print(f"{conflict_type:35s} {count:4d} ({percentage:5.1f}%)")


def main():
    """Main"""
    parser = argparse.ArgumentParser(description="Evaluate reasoning agent.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model id from evaluation/openai-apis/custom_models.json. "
             "If omitted, uses the first model in that file."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N policies after --start. If omitted, evaluate all."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based) before applying --limit."
    )
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATING REASONING AGENT")
    print("="*80)
    
    # Load policies
    rejected_file = Path("data/rejected_policies/rejected_policies_dataset.json")
    approved_file = Path("data/approved_policies/approved_policies_dataset.json")
    
    policies = []
    
    if rejected_file.exists():
        with open(rejected_file, 'r') as f:
            data = json.load(f)
            policies.extend(data["policies"])
            print(f"Loaded {len(data['policies'])} REJECTED")
    
    if approved_file.exists():
        with open(approved_file, 'r') as f:
            data = json.load(f)
            policies.extend(data["policies"])
            print(f"Loaded {len(data['policies'])} APPROVED")

    full_total = len(policies)
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0 when provided")

    if args.start > full_total:
        policies = []
    elif args.limit is None:
        policies = policies[args.start:]
    else:
        policies = policies[args.start:args.start + args.limit]

    print(f"\n Total loaded: {full_total}")
    print(f" Evaluating:   {len(policies)} (start={args.start}, limit={args.limit})")
    
    # Load model config
    model_config = load_model_config(args.model_id)
    print("\n Model config:")
    print(f"   model_id: {model_config['model_id']}")
    print(f"   base_url: {model_config['base_url']}")

    # Initialize agent
    print("\n Initializing agent...")
    reasoner = Reasoner(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        model=model_config["model_id"],
        temperature=0.0
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("RUNNING...")
    print("="*80)
    
    results = []
    for i, policy in enumerate(policies, 1):
        print(f"{i}/{len(policies)}: {policy['policy_id'][:40]}...", end='\r')
        result = evaluate_policy(reasoner, policy)
        results.append(result)
    
    print()
    
    # Metrics
    metrics = calculate_metrics(results)
    
    # Save
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "agent_results.json", 'w') as f:
        json.dump([vars(r) for r in results], f, indent=2)
    
    # Print
    print_results(metrics)
    
    # Show missed
    missed = [
        r for r in results
        if r.expected_binary == "REJECTED" and r.agent_decision != "reject"
    ]
    if missed:
        print(f"\nMISSED {len(missed)} BAD POLICIES:")
        for r in missed[:5]:
            print(f"   - {r.policy_id}")
    
    print(f"\nDone! Saved to evaluation/results/agent_results.json")


if __name__ == "__main__":
    main()