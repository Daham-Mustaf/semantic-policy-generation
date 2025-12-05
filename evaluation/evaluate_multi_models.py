# evaluation/evaluate_multi_models.py

"""
Evaluate Multiple Models on ODRL Policy Reasoning
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import sys
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.reasoner.reasoner_agent import Reasoner


@dataclass
class SimpleResult:
    policy_id: str
    expected: str
    agent_decision: str
    correct: bool


@dataclass
class SimpleMetrics:
    model_name: str
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
    policy_id = policy["policy_id"]
    policy_text = policy["policy_text"]
    expected = policy["ground_truth"]["expected_outcome"]
    
    try:
        result = reasoner.reason(policy_text)
        agent_decision = result.get("decision", "needs_input").lower()
        
        if expected == "APPROVED":
            correct = (agent_decision == "approve")
        else:
            correct = (agent_decision == "reject")
        
        return SimpleResult(
            policy_id=policy_id,
            expected=expected,
            agent_decision=agent_decision,
            correct=correct
        )
    except Exception as e:
        print(f"\n⚠️  Error: {policy_id}: {str(e)[:100]}")
        return SimpleResult(
            policy_id=policy_id,
            expected=expected,
            agent_decision="ERROR",
            correct=False
        )


def calculate_metrics(model_name: str, results: List[SimpleResult]) -> SimpleMetrics:
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    
    should_reject = [r for r in results if r.expected == "REJECTED"]
    should_approve = [r for r in results if r.expected == "APPROVED"]
    
    correctly_rejected = sum(1 for r in should_reject if r.agent_decision == "reject")
    missed = sum(1 for r in should_reject if r.agent_decision == "approve")
    
    correctly_approved = sum(1 for r in should_approve if r.agent_decision == "approve")
    over_rejected = sum(1 for r in should_approve if r.agent_decision == "reject")
    
    return SimpleMetrics(
        model_name=model_name,
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


def evaluate_model(model_name: str, model_id: str, policies: List[dict], base_url: str, api_key: str) -> SimpleMetrics:
    """Evaluate a single model"""
    
    print(f"\n{'='*80}")
    print(f"🤖 EVALUATING {model_name}")
    print(f"{'='*80}")
    
    # Initialize agent
    reasoner = Reasoner(
        api_key=api_key,
        base_url=base_url,
        model=model_id,
        temperature=0.0
    )
    
    # Evaluate all policies
    results = []
    for i, policy in enumerate(policies, 1):
        print(f"{i}/{len(policies)}: {policy['policy_id'][:50]}...", end='\r')
        result = evaluate_policy(reasoner, policy)
        results.append(result)
    
    print()
    
    # Calculate metrics
    metrics = calculate_metrics(model_name, results)
    
    # Save results
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = model_name.lower().replace(' ', '_').replace(':', '_').replace('-', '_')
    
    with open(output_dir / f"{safe_name}_results.json", 'w', encoding='utf-8') as f:
        json.dump([vars(r) for r in results], f, indent=2)
    
    with open(output_dir / f"{safe_name}_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(vars(metrics), f, indent=2)
    
    print(f"✓ Saved to evaluation/results/{safe_name}_*.json")
    
    return metrics


def print_comparison(all_metrics: List[SimpleMetrics]):
    """Print comparison table"""
    
    print("\n" + "="*100)
    print("MULTI-MODEL COMPARISON")
    print("="*100)
    print(f"{'Model':<25} {'Overall':<10} {'Rejection':<12} {'Approval':<12} {'Missed':<8} {'Over-Rej':<10}")
    print("-"*100)
    
    for m in all_metrics:
        print(f"{m.model_name:<25} {m.accuracy:>7.1f}%  {m.rejection_accuracy:>9.1f}%  {m.approval_accuracy:>9.1f}%  "
              f"{m.missed:<8} {m.over_rejected:<10}")


def main():
    print("="*80)
    print(" MULTI-MODEL EVALUATION")
    print("="*80)
    
    # Load config
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    
    print(f"\n🔧 Configuration:")
    print(f"   Base URL: {base_url}")
    
    # Load policies once
    rejected_file = Path("data/rejected_policies/rejected_policies_unified.json")
    approved_file = Path("data/approved_policies/approved_policies_unified.json")
    
    policies = []
    
    if rejected_file.exists():
        with open(rejected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            policies.extend(data["policies"])
            print(f"   ✓ Loaded {len(data['policies'])} REJECTED")
    
    if approved_file.exists():
        with open(approved_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            policies.extend(data["policies"])
            print(f"   ✓ Loaded {len(data['policies'])} APPROVED")
    
    print(f"\n📊 Total: {len(policies)} policies")
    
    # Models to test
    models_to_test = [
        # {"name": "DeepSeek R1 70B", "id": "deepseek-r1:70b"},
        {"name": "GPT-OSS 120B", "id": "gpt-oss:120b"},
        {"name": "Llama 3.3 70B", "id": "llama3.3:70b"},
        # {"name": "Llama 3.1 70B", "id": "llama3.1:70b"},
    ]
    
    print(f"\n🤖 Testing {len(models_to_test)} models:")
    for model in models_to_test:
        print(f"   • {model['name']}")
    
    # Evaluate each model
    all_metrics = []
    
    for model_config in models_to_test:
        try:
            metrics = evaluate_model(
                model_name=model_config["name"],
                model_id=model_config["id"],
                policies=policies,
                base_url=base_url,
                api_key=api_key
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n Failed to evaluate {model_config['name']}: {e}")
            continue
    
    # Print comparison
    if all_metrics:
        print_comparison(all_metrics)
        
        # Save comparison
        output_dir = Path("evaluation/results")
        with open(output_dir / "multi_model_comparison.json", 'w', encoding='utf-8') as f:
            json.dump([vars(m) for m in all_metrics], f, indent=2)
        
        print(f"\nComplete! Saved to evaluation/results/multi_model_comparison.json")


if __name__ == "__main__":
    main()