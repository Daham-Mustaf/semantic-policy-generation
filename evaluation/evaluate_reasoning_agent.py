# evaluation/evaluate_reasoning_agent.py

"""
Evaluate YOUR Reasoning Agent on the unified dataset
Tests the 6-phase structured conflict detection
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from agents.reasoner.reasoner_agent import Reasoner


@dataclass
class SimpleResult:
    """Simple result"""
    policy_id: str
    expected: str  # "APPROVED" or "REJECTED"
    agent_decision: str  # "approve", "reject", "needs_input"
    correct: bool


@dataclass
class SimpleMetrics:
    """Simple metrics"""
    total: int
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


def evaluate_policy(reasoner: Reasoner, policy: dict) -> SimpleResult:
    """Evaluate one policy"""
    
    policy_id = policy["policy_id"]
    policy_text = policy["policy_text"]
    expected = policy["ground_truth"]["expected_outcome"]
    
    try:
        # Call agent
        result = reasoner.reason(policy_text)
        agent_decision = result.get("decision", "needs_input").lower()
        
        # Check correctness
        if expected == "APPROVED":
            correct = (agent_decision == "approve")
        else:  # "REJECTED"
            correct = (agent_decision == "reject")
        
        return SimpleResult(
            policy_id=policy_id,
            expected=expected,
            agent_decision=agent_decision,
            correct=correct
        )
        
    except Exception as e:
        print(f"\n Error: {policy_id}: {str(e)[:100]}")
        return SimpleResult(
            policy_id=policy_id,
            expected=expected,
            agent_decision="ERROR",
            correct=False
        )


def calculate_metrics(results: List[SimpleResult]) -> SimpleMetrics:
    """Calculate metrics"""
    
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    
    should_reject = [r for r in results if r.expected == "REJECTED"]
    should_approve = [r for r in results if r.expected == "APPROVED"]
    
    correctly_rejected = sum(1 for r in should_reject if r.agent_decision == "reject")
    missed = sum(1 for r in should_reject if r.agent_decision == "approve")
    
    correctly_approved = sum(1 for r in should_approve if r.agent_decision == "approve")
    over_rejected = sum(1 for r in should_approve if r.agent_decision == "reject")
    
    return SimpleMetrics(
        total=total,
        correct=correct,
        accuracy=(correct / total * 100) if total > 0 else 0,
        should_reject=len(should_reject),
        correctly_rejected=correctly_rejected,
        missed=missed,
        rejection_accuracy=(correctly_rejected / len(should_reject) * 100) if should_reject else 0,
        should_approve=len(should_approve),
        correctly_approved=correctly_approved,
        over_rejected=over_rejected,
        approval_accuracy=(correctly_approved / len(should_approve) * 100) if should_approve else 0
    )


def print_results(metrics: SimpleMetrics):
    """Print results"""
    
    print("\n" + "="*80)
    print("REASONING AGENT RESULTS")
    print("="*80)
    print(f"Total:             {metrics.total}")
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


def main():
    """Main"""
    
    print("="*80)
    print("EVALUATING REASONING AGENT")
    print("="*80)
    
    # Load policies
    rejected_file = Path("data/rejected_policies/rejected_policies_unified.json")
    approved_file = Path("data/approved_policies/approved_policies_unified.json")
    
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
    
    print(f"\n Total: {len(policies)}")
    
    # Initialize agent
    print("\n Initializing agent...")
    reasoner = Reasoner(
        api_key="xxx",
        api_version="2024-10-01-preview",
        azure_endpoint="https://fhgenie-api-fit-ems30127.openai.azure.com/",
        model="gpt-4o-2024-11-20",
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
    missed = [r for r in results if r.expected == "REJECTED" and r.agent_decision == "approve"]
    if missed:
        print(f"\nMISSED {len(missed)} BAD POLICIES:")
        for r in missed[:5]:
            print(f"   - {r.policy_id}")
    
    print(f"\nDone! Saved to evaluation/results/agent_results.json")


if __name__ == "__main__":
    main()