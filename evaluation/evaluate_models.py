# evaluation/evaluate_models.py
"""
Multi-Model ODRL Generation Evaluation
Compares different LLMs on reasoning, generation, and validation quality
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage
from openai import AzureOpenAI 


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.reasoner.reasoner_agent import Reasoner
# evaluation/evaluate_models.py (FINAL CORRECTED VERSION)


@dataclass
class SimpleResult:
    """Simple policy evaluation result"""
    policy_id: str
    expected_outcome: str  # "APPROVED" or "REJECTED"
    model_decision: str    # "APPROVE", "REJECT", or "ERROR"
    correct: bool


@dataclass
class SimpleMetrics:
    """Simple aggregated metrics"""
    model_name: str
    total_policies: int
    correct_decisions: int
    accuracy: float
    
    # Rejection metrics
    should_reject: int
    correctly_rejected: int
    missed_rejections: int
    rejection_accuracy: float
    
    # Approval metrics
    should_approve: int
    correctly_approved: int
    false_rejections: int
    approval_accuracy: float
    
    errors: int


def call_llm_direct(client: AzureOpenAI, model: str, policy_text: str) -> str:
    """
    Call LLM directly with simple prompt
    Returns: "APPROVE" or "REJECT" or "ERROR"
    """
    
    prompt = f"""You are an ODRL policy conflict detector. Analyze this policy and decide whether to APPROVE or REJECT it.

REJECT if the policy contains ANY conflicts, contradictions, or unmeasurable terms.
APPROVE if the policy is clear, consistent, and enforceable.

Respond with ONLY ONE WORD: either "APPROVE" or "REJECT"

Policy:
{policy_text}

Decision:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an ODRL policy validator. Respond with only 'APPROVE' or 'REJECT'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        decision = response.choices[0].message.content.strip().upper()
        
        # Clean up response
        if "APPROVE" in decision:
            return "APPROVE"
        elif "REJECT" in decision:
            return "REJECT"
        else:
            return "ERROR"
            
    except Exception as e:
        print(f"\n❌ LLM Error: {e}")
        return "ERROR"


def evaluate_single_policy(client: AzureOpenAI, model: str, policy: dict) -> SimpleResult:
    """Evaluate a single policy"""
    
    policy_id = policy["policy_id"]
    policy_text = policy["policy_text"]
    expected = policy["ground_truth"]["expected_outcome"]  # "APPROVED" or "REJECTED"
    
    # Get LLM decision
    decision = call_llm_direct(client, model, policy_text)
    
    # Check correctness
    if expected == "APPROVED":
        correct = (decision == "APPROVE")
    else:  # "REJECTED"
        correct = (decision == "REJECT")
    
    return SimpleResult(
        policy_id=policy_id,
        expected_outcome=expected,
        model_decision=decision,
        correct=correct
    )


def calculate_metrics(model_name: str, results: List[SimpleResult]) -> SimpleMetrics:
    """Calculate simple metrics"""
    
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    errors = sum(1 for r in results if r.model_decision == "ERROR")
    
    # Breakdown
    should_reject = [r for r in results if r.expected_outcome == "REJECTED"]
    should_approve = [r for r in results if r.expected_outcome == "APPROVED"]
    
    correctly_rejected = sum(1 for r in should_reject if r.model_decision == "REJECT")
    missed_rejections = sum(1 for r in should_reject if r.model_decision == "APPROVE")
    
    correctly_approved = sum(1 for r in should_approve if r.model_decision == "APPROVE")
    false_rejections = sum(1 for r in should_approve if r.model_decision == "REJECT")
    
    return SimpleMetrics(
        model_name=model_name,
        total_policies=total,
        correct_decisions=correct,
        accuracy=(correct / total * 100) if total > 0 else 0,
        
        should_reject=len(should_reject),
        correctly_rejected=correctly_rejected,
        missed_rejections=missed_rejections,
        rejection_accuracy=(correctly_rejected / len(should_reject) * 100) if should_reject else 0,
        
        should_approve=len(should_approve),
        correctly_approved=correctly_approved,
        false_rejections=false_rejections,
        approval_accuracy=(correctly_approved / len(should_approve) * 100) if should_approve else 0,
        
        errors=errors
    )


def print_results(all_metrics: List[SimpleMetrics]):
    """Print comparison table"""
    
    print("\n" + "="*100)
    print("MODEL COMPARISON - SIMPLE REJECTION TEST")
    print("="*100)
    print(f"{'Model':<15} {'Total':<7} {'Correct':<8} {'Accuracy':<12} {'Errors':<7}")
    print("-"*100)
    
    for m in all_metrics:
        print(f"{m.model_name:<15} {m.total_policies:<7} {m.correct_decisions:<8} "
              f"{m.accuracy:>8.1f}%    {m.errors:<7}")
    
    print("\n" + "="*100)
    print("REJECTION TEST (Should Reject → Did Reject)")
    print("="*100)
    print(f"{'Model':<15} {'Should Reject':<14} {'✓ Rejected':<12} {'✗ Missed':<10} {'Accuracy':<12}")
    print("-"*100)
    
    for m in all_metrics:
        print(f"{m.model_name:<15} {m.should_reject:<14} {m.correctly_rejected:<12} "
              f"{m.missed_rejections:<10} {m.rejection_accuracy:>8.1f}%")
    
    print("\n" + "="*100)
    print("APPROVAL TEST (Should Approve → Did Approve)")
    print("="*100)
    print(f"{'Model':<15} {'Should Approve':<15} {'✓ Approved':<12} {'✗ Rejected':<12} {'Accuracy':<12}")
    print("-"*100)
    
    for m in all_metrics:
        print(f"{m.model_name:<15} {m.should_approve:<15} {m.correctly_approved:<12} "
              f"{m.false_rejections:<12} {m.approval_accuracy:>8.1f}%")


def main():
    """Main evaluation"""
    
    print("="*100)
    print("ULTRA-SIMPLE MODEL EVALUATION")
    print("="*100)
    
    # Load datasets
    rejected_file = Path("data/rejected_policies/rejected_policies_unified.json")
    approved_file = Path("data/approved_policies/approved_policies_unified.json")
    
    policies = []
    
    if rejected_file.exists():
        with open(rejected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            policies.extend(data["policies"])
            print(f"✓ Loaded {len(data['policies'])} REJECTED policies")
    
    if approved_file.exists():
        with open(approved_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            policies.extend(data["policies"])
            print(f"✓ Loaded {len(data['policies'])} APPROVED policies")
    
    print(f"\n Total: {len(policies)} policies")
    
    # Setup Azure OpenAI
    client = AzureOpenAI(
        api_key="xx",
        api_version="2024-10-01-preview",
        azure_endpoint="https://fhgenie-api-fit-ems30127.openai.azure.com/"
    )
    
    # Models to test
    models = [
        {"name": "GPT-4o", "model": "gpt-4o-2024-11-20"},
        {"name": "GPT-4o-mini", "model": "gpt-4o-mini"},
        {"name": "GPT-4-Turbo", "model": "gpt-4-turbo-2024-04-09"}
    ]
    
    all_metrics = []
    
    # Evaluate each model
    for model_config in models:
        print(f"\n{'='*100}")
        print(f"Evaluating {model_config['name']}")
        print(f"{'='*100}")
        
        results = []
        for i, policy in enumerate(policies, 1):
            print(f"Processing {i}/{len(policies)}: {policy['policy_id'][:50]}...", end='\r')
            result = evaluate_single_policy(client, model_config["model"], policy)
            results.append(result)
        
        print()  # New line
        
        # Calculate metrics
        metrics = calculate_metrics(model_config["name"], results)
        all_metrics.append(metrics)
        
        # Save results
        output_dir = Path("evaluation/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{model_config['name'].lower().replace('-', '_')}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([vars(r) for r in results], f, indent=2)
        
        print(f"✓ Saved to {output_file}")
    
    # Print comparison
    print_results(all_metrics)
    
    # Save metrics
    metrics_file = Path("evaluation/results/comparison.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump([vars(m) for m in all_metrics], f, indent=2)
    
    print(f"\n✅ Complete! Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()