
# evaluation/evaluate_deepseek.py

"""
Evaluate DeepSeek R1 70B on ODRL Policy Reasoning
Tests the 6-phase structured conflict detection using local FIT server
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List
import sys
from dotenv import load_dotenv

# Load environment variables

load_dotenv()
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
        print(f"\n⚠️ Error: {policy_id}: {str(e)[:100]}")
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
    print("🤖 DEEPSEEK R1 70B - REASONING AGENT RESULTS")
    print("="*80)
    print(f"Total Policies:    {metrics.total}")
    print(f"Correct Decisions: {metrics.correct}")
    print(f"Overall Accuracy:  {metrics.accuracy:.1f}%")
    
    print("\n" + "="*80)
    print("🚫 REJECTION TEST (Catching Bad Policies)")
    print("="*80)
    print(f"Should Reject:     {metrics.should_reject}")
    print(f"✅ Correctly Rejected:  {metrics.correctly_rejected}")
    print(f"❌ Missed Rejections:   {metrics.missed}")
    print(f"Rejection Accuracy:     {metrics.rejection_accuracy:.1f}%")
    
    print("\n" + "="*80)
    print("✅ APPROVAL TEST (Not Over-rejecting Good Policies)")
    print("="*80)
    print(f"Should Approve:    {metrics.should_approve}")
    print(f"✅ Correctly Approved:  {metrics.correctly_approved}")
    print(f"❌ Over-rejected:       {metrics.over_rejected}")
    print(f"Approval Accuracy:      {metrics.approval_accuracy:.1f}%")


def main():
    """Main"""
    
    print("="*80)
    print("🚀 EVALUATING DEEPSEEK R1 70B ON ODRL POLICY REASONING")
    print("="*80)
    
    # Load environment variables
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    
    print(f"\n🔧 Configuration:")
    print(f"   Base URL: {base_url}")
    print(f"   Model:    {model}")
    
    # Load policies
    rejected_file = Path("data/rejected_policies/rejected_policies_unified.json")
    approved_file = Path("data/approved_policies/approved_policies_unified.json")
    
    policies = []
    
    if rejected_file.exists():
        with open(rejected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            policies.extend(data["policies"])
            print(f"   ✓ Loaded {len(data['policies'])} REJECTED policies")
    
    if approved_file.exists():
        with open(approved_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            policies.extend(data["policies"])
            print(f"   ✓ Loaded {len(data['policies'])} APPROVED policies")
    
    print(f"\n📊 Total: {len(policies)} policies")
    
    # Initialize agent with DeepSeek
    print("\n🤖 Initializing DeepSeek R1 70B agent...")
    
    reasoner = Reasoner(
        api_key=api_key,
        azure_endpoint=base_url,
        model=model,
        temperature=0.0
    )
    
    # Evaluate
    print("\n" + "="*80)
    print("⚙️  RUNNING EVALUATION...")
    print("="*80)
    
    results = []
    for i, policy in enumerate(policies, 1):
        print(f"Processing {i}/{len(policies)}: {policy['policy_id'][:50]}...", end='\r')
        result = evaluate_policy(reasoner, policy)
        results.append(result)
    
    print()
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Save results
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "deepseek_r1_70b_results.json", 'w', encoding='utf-8') as f:
        json.dump([vars(r) for r in results], f, indent=2)
    
    # Save metrics
    with open(output_dir / "deepseek_r1_70b_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(vars(metrics), f, indent=2)
    
    # Print results
    print_results(metrics)
    
    # Show examples of missed policies
    missed = [r for r in results if r.expected == "REJECTED" and r.agent_decision == "approve"]
    if missed:
        print(f"\nMISSED {len(missed)} BAD POLICIES (First 5):")
        for r in missed[:5]:
            print(f"   • {r.policy_id}")
    
    # Show examples of over-rejected policies
    over_rejected = [r for r in results if r.expected == "APPROVED" and r.agent_decision == "reject"]
    if over_rejected:
        print(f"\n OVER-REJECTED {len(over_rejected)} GOOD POLICIES (First 5):")
        for r in over_rejected[:5]:
            print(f"   • {r.policy_id}")
    
    print(f"\nEvaluation Complete!")
    print(f"Results saved to:")
    print(f"   • evaluation/results/deepseek_r1_70b_results.json")
    print(f"   • evaluation/results/deepseek_r1_70b_metrics.json")


if __name__ == "__main__":
    main()