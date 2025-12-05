# evaluation/evaluate_pipeline.py

"""
End-to-End Pipeline Evaluation
Tests: Reasoner → Generator → Validator
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import sys
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.reasoner.reasoner_agent import Reasoner
from agents.generator.generator import Generator
from agents.validator.validator_agent import ValidatorAgent


@dataclass
class PipelineResult:
    """Result for one policy through full pipeline"""
    policy_id: str
    policy_text: str
    expected_outcome: str  # "APPROVED" or "REJECTED"
    
    # Reasoner stage
    reasoner_decision: str  # "approve", "reject", "needs_input"
    reasoner_correct: bool
    
    # Generator stage (only if approved)
    generator_ran: bool
    odrl_generated: bool
    odrl_turtle: str = ""
    
    # Validator stage (only if generated)
    validator_ran: bool
    validation_passed: bool
    validation_attempts: int = 0
    
    # Overall success
    pipeline_success: bool  # True if: approved → generated → validated
    

@dataclass
class PipelineMetrics:
    """Aggregated pipeline metrics"""
    model_name: str
    total_policies: int
    
    # Reasoner metrics
    reasoner_correct: int
    reasoner_accuracy: float
    correctly_approved: int
    incorrectly_rejected: int
    
    # Generator metrics
    generator_attempts: int
    odrl_generated: int
    generation_success_rate: float
    
    # Validator metrics
    validator_attempts: int
    first_attempt_valid: int
    final_valid: int
    validation_success_rate: float
    avg_regen_attempts: float
    
    # End-to-end metrics
    end_to_end_success: int
    end_to_end_success_rate: float


def evaluate_pipeline_single(
    reasoner: Reasoner,
    generator: Generator,
    validator: ValidatorAgent,
    policy: dict
) -> PipelineResult:
    """Evaluate one policy through full pipeline"""
    
    policy_id = policy["policy_id"]
    policy_text = policy["policy_text"]
    expected = policy["ground_truth"]["expected_outcome"]
    
    result = PipelineResult(
        policy_id=policy_id,
        policy_text=policy_text,
        expected_outcome=expected,
        reasoner_decision="ERROR",
        reasoner_correct=False,
        generator_ran=False,
        odrl_generated=False,
        validator_ran=False,
        validation_passed=False,
        pipeline_success=False
    )
    
    # ===== STAGE 1: REASONER =====
    try:
        reasoner_result = reasoner.reason(policy_text)
        result.reasoner_decision = reasoner_result.get("decision", "needs_input").lower()
        
        # Check if reasoner is correct
        if expected == "APPROVED":
            result.reasoner_correct = (result.reasoner_decision == "approve")
        else:  # REJECTED
            result.reasoner_correct = (result.reasoner_decision == "reject")
        
        # Only continue if approved
        if result.reasoner_decision != "approve":
            return result
            
    except Exception as e:
        print(f"\n⚠️  Reasoner error {policy_id}: {str(e)[:80]}")
        return result
    
    # ===== STAGE 2: GENERATOR =====
    try:
        result.generator_ran = True
        gen_result = generator.generate(policy_text, policy_id)
        result.odrl_turtle = gen_result["odrl_turtle"]
        result.odrl_generated = True
        
    except Exception as e:
        print(f"\n⚠️  Generator error {policy_id}: {str(e)[:80]}")
        return result
    
    # ===== STAGE 3: VALIDATOR =====
    try:
        result.validator_ran = True
        
        # Validate with up to 3 regeneration attempts
        validation_result = validator.validate_and_regenerate(
            policy_text=policy_text,
            odrl_turtle=result.odrl_turtle,
            max_attempts=3
        )
        
        result.validation_passed = validation_result["success"]
        result.validation_attempts = validation_result["attempts"]
        
        if validation_result["success"]:
            result.odrl_turtle = validation_result["final_odrl"]
            result.pipeline_success = True
        
    except Exception as e:
        print(f"\n⚠️  Validator error {policy_id}: {str(e)[:80]}")
        return result
    
    return result


def calculate_pipeline_metrics(model_name: str, results: List[PipelineResult]) -> PipelineMetrics:
    """Calculate comprehensive pipeline metrics"""
    
    total = len(results)
    
    # Filter to policies that SHOULD be approved
    should_approve = [r for r in results if r.expected_outcome == "APPROVED"]
    
    # Reasoner metrics
    reasoner_correct = sum(1 for r in results if r.reasoner_correct)
    correctly_approved = sum(1 for r in should_approve if r.reasoner_decision == "approve")
    incorrectly_rejected = sum(1 for r in should_approve if r.reasoner_decision != "approve")
    
    # Generator metrics
    generator_ran = [r for r in results if r.generator_ran]
    odrl_generated = sum(1 for r in generator_ran if r.odrl_generated)
    
    # Validator metrics
    validator_ran = [r for r in results if r.validator_ran]
    first_attempt_valid = sum(1 for r in validator_ran if r.validation_passed and r.validation_attempts == 1)
    final_valid = sum(1 for r in validator_ran if r.validation_passed)
    
    # Average regeneration attempts (only for policies that needed regen)
    needed_regen = [r for r in validator_ran if r.validation_attempts > 1]
    avg_regen = sum(r.validation_attempts for r in needed_regen) / len(needed_regen) if needed_regen else 0
    
    # End-to-end success
    e2e_success = sum(1 for r in should_approve if r.pipeline_success)
    
    return PipelineMetrics(
        model_name=model_name,
        total_policies=total,
        
        reasoner_correct=reasoner_correct,
        reasoner_accuracy=(reasoner_correct / total * 100) if total > 0 else 0,
        correctly_approved=correctly_approved,
        incorrectly_rejected=incorrectly_rejected,
        
        generator_attempts=len(generator_ran),
        odrl_generated=odrl_generated,
        generation_success_rate=(odrl_generated / len(generator_ran) * 100) if generator_ran else 0,
        
        validator_attempts=len(validator_ran),
        first_attempt_valid=first_attempt_valid,
        final_valid=final_valid,
        validation_success_rate=(final_valid / len(validator_ran) * 100) if validator_ran else 0,
        avg_regen_attempts=avg_regen,
        
        end_to_end_success=e2e_success,
        end_to_end_success_rate=(e2e_success / len(should_approve) * 100) if should_approve else 0
    )


def print_pipeline_results(metrics: PipelineMetrics):
    """Print pipeline evaluation results"""
    
    print("\n" + "="*100)
    print(f"🔄 END-TO-END PIPELINE RESULTS - {metrics.model_name}")
    print("="*100)
    
    print("\n📊 OVERALL METRICS")
    print("-"*100)
    print(f"Total Policies Evaluated:  {metrics.total_policies}")
    print(f"Reasoner Accuracy:         {metrics.reasoner_accuracy:.1f}%")
    print(f"End-to-End Success Rate:   {metrics.end_to_end_success_rate:.1f}%")
    
    print("\n" + "="*100)
    print("🤔 STAGE 1: REASONER (Conflict Detection)")
    print("="*100)
    print(f"Correct Decisions:         {metrics.reasoner_correct}/{metrics.total_policies}")
    print(f"Correctly Approved:        {metrics.correctly_approved}")
    print(f"Incorrectly Rejected:      {metrics.incorrectly_rejected}")
    print(f"Accuracy:                  {metrics.reasoner_accuracy:.1f}%")
    
    print("\n" + "="*100)
    print("⚙️  STAGE 2: GENERATOR (ODRL Creation)")
    print("="*100)
    print(f"Generation Attempts:       {metrics.generator_attempts}")
    print(f"Successfully Generated:    {metrics.odrl_generated}")
    print(f"Success Rate:              {metrics.generation_success_rate:.1f}%")
    
    print("\n" + "="*100)
    print("✅ STAGE 3: VALIDATOR (SHACL Conformance)")
    print("="*100)
    print(f"Validation Attempts:       {metrics.validator_attempts}")
    print(f"Valid on First Attempt:    {metrics.first_attempt_valid}")
    print(f"Valid After Regeneration:  {metrics.final_valid - metrics.first_attempt_valid}")
    print(f"Final Valid:               {metrics.final_valid}")
    print(f"Validation Success Rate:   {metrics.validation_success_rate:.1f}%")
    print(f"Avg Regen Attempts:        {metrics.avg_regen_attempts:.2f}")
    
    print("\n" + "="*100)
    print("🎯 END-TO-END SUCCESS")
    print("="*100)
    print(f"Policies reaching final stage: {metrics.validator_attempts}")
    print(f"Successfully completed:        {metrics.end_to_end_success}")
    print(f"E2E Success Rate:              {metrics.end_to_end_success_rate:.1f}%")
    
    # Calculate cost efficiency
    if metrics.avg_regen_attempts > 0:
        cost_savings = ((3 - metrics.avg_regen_attempts) / 3 * 100)
        print(f"\n💰 Cost Efficiency:")
        print(f"   Avg regenerations: {metrics.avg_regen_attempts:.2f} (vs 3 max)")
        print(f"   Cost savings: ~{cost_savings:.1f}% vs always using max attempts")


def main():
    print("="*100)
    print("🚀 END-TO-END PIPELINE EVALUATION")
    print("="*100)
    
    # Load config
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    
    # For Azure (GPT-4o), use these instead:
    use_azure = True  # Set to False for Fraunhofer models
    
    if use_azure:
        azure_config = {
            "api_key": "xx",
            "api_version": "2024-10-01-preview",
            "azure_endpoint": "https://fhgenie-api-fit-ems30127.openai.azure.com/",
            "model": "gpt-4o-2024-11-20"
        }
        model_name = "GPT-4o (Azure)"
    else:
        azure_config = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model
        }
        model_name = model
    
    print(f"\n🔧 Configuration:")
    print(f"   Model: {model_name}")
    
    # Load ONLY approved policies (since we're testing generation)
    approved_file = Path("data/approved_policies/approved_policies_unified.json")
    
    if not approved_file.exists():
        print(f"\n❌ Error: {approved_file} not found")
        return
    
    with open(approved_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        policies = data["policies"][:5]  # Test with first 5 policies
        print(f"   ✓ Loaded {len(policies)} APPROVED policies (testing subset)")
    
    # Initialize all three agents
    print(f"\n🤖 Initializing pipeline agents...")
    
    reasoner = Reasoner(**azure_config, temperature=0.0)
    generator = Generator(**azure_config, temperature=0.0)
    validator = ValidatorAgent(**azure_config, temperature=0.0)
    
    print("   ✓ Reasoner initialized")
    print("   ✓ Generator initialized")
    print("   ✓ Validator initialized")
    
    # Run pipeline evaluation
    print(f"\n" + "="*100)
    print("⚙️  RUNNING PIPELINE EVALUATION...")
    print("="*100)
    
    results = []
    for i, policy in enumerate(policies, 1):
        print(f"\n[{i}/{len(policies)}] Processing: {policy['policy_id'][:60]}")
        print("-" * 100)
        
        result = evaluate_pipeline_single(reasoner, generator, validator, policy)
        results.append(result)
        
        # Print immediate result
        print(f"   Reasoner: {result.reasoner_decision.upper():<10} "
              f"({'✓' if result.reasoner_correct else '✗'})")
        if result.generator_ran:
            print(f"   Generator: {'SUCCESS' if result.odrl_generated else 'FAILED':<10}")
        if result.validator_ran:
            print(f"   Validator: {'VALID' if result.validation_passed else 'INVALID':<10} "
                  f"(attempts: {result.validation_attempts})")
        print(f"   Pipeline: {'SUCCESS' if result.pipeline_success else '❌ FAILED'}")
    
    # Calculate metrics
    metrics = calculate_pipeline_metrics(model_name, results)
    
    # Save results
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = model_name.lower().replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '')
    
    # Save detailed results
    with open(output_dir / f"{safe_name}_pipeline_results.json", 'w', encoding='utf-8') as f:
        json.dump([{
            "policy_id": r.policy_id,
            "expected_outcome": r.expected_outcome,
            "reasoner_decision": r.reasoner_decision,
            "reasoner_correct": r.reasoner_correct,
            "generator_ran": r.generator_ran,
            "odrl_generated": r.odrl_generated,
            "validator_ran": r.validator_ran,
            "validation_passed": r.validation_passed,
            "validation_attempts": r.validation_attempts,
            "pipeline_success": r.pipeline_success,
            "odrl_turtle": r.odrl_turtle[:500] if r.odrl_turtle else ""  # Truncate for readability
        } for r in results], f, indent=2)
    
    # Save metrics
    with open(output_dir / f"{safe_name}_pipeline_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(vars(metrics), f, indent=2)
    
    # Print results
    print_pipeline_results(metrics)
    
    print(f"\n✅ Complete! Results saved to:")
    print(f"   • {output_dir / f'{safe_name}_pipeline_results.json'}")
    print(f"   • {output_dir / f'{safe_name}_pipeline_metrics.json'}")


if __name__ == "__main__":
    main()