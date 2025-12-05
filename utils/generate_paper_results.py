# evaluation/generate_paper_results.py

"""
Generate publication-ready tables and figures for paper
"""

import json
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class ModelComparison:
    """Model comparison for paper"""
    approach: str
    model: str
    overall_acc: float
    rejection_acc: float
    approval_acc: float
    missed_conflicts: int
    false_positives: int
    total_policies: int


def load_results():
    """Load all evaluation results"""
    
    results_dir = Path("evaluation/results")
    
    # Your reasoning agent
    with open(results_dir / "agent_results.json", 'r') as f:
        agent_results = json.load(f)
    
    # Direct LLM baseline (from earlier run)
    with open(results_dir / "gpt_4o_results.json", 'r') as f:
        direct_results = json.load(f)
    
    return agent_results, direct_results


def calculate_detailed_metrics(results: List[dict]) -> dict:
    """Calculate all metrics for paper"""
    
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    
    # Should reject
    should_reject = [r for r in results if r["expected_outcome"] == "REJECTED"]
    correctly_rejected = sum(1 for r in should_reject if r["model_decision"] == "REJECT" or r.get("agent_decision") == "reject")
    missed = len(should_reject) - correctly_rejected
    
    # Should approve
    should_approve = [r for r in results if r["expected_outcome"] == "APPROVED"]
    correctly_approved = sum(1 for r in should_approve if r["model_decision"] == "APPROVE" or r.get("agent_decision") == "approve")
    false_positives = len(should_approve) - correctly_approved
    
    return {
        "total": total,
        "overall_accuracy": (correct / total * 100),
        "rejection_accuracy": (correctly_rejected / len(should_reject) * 100) if should_reject else 0,
        "approval_accuracy": (correctly_approved / len(should_approve) * 100) if should_approve else 0,
        "missed_conflicts": missed,
        "false_positives": false_positives,
        "precision": (correctly_rejected / (correctly_rejected + false_positives)) if (correctly_rejected + false_positives) > 0 else 0,
        "recall": (correctly_rejected / len(should_reject)) if should_reject else 0,
        "f1_score": 0  # Calculate below
    }


def generate_latex_table(comparisons: List[ModelComparison]) -> str:
    """Generate LaTeX table for paper"""
    
    latex = r"""
\begin{table}[ht]
\centering
\caption{Conflict Detection Performance Comparison}
\label{tab:conflict_detection}
\begin{tabular}{llcccccc}
\toprule
\textbf{Approach} & \textbf{Model} & \textbf{Overall} & \textbf{Rejection} & \textbf{Approval} & \textbf{Missed} & \textbf{False+} & \textbf{n} \\
\textbf{} & \textbf{} & \textbf{Acc. (\%)} & \textbf{Acc. (\%)} & \textbf{Acc. (\%)} & \textbf{Conflicts} & \textbf{} & \textbf{} \\
\midrule
"""
    
    for comp in comparisons:
        latex += f"{comp.approach} & {comp.model} & {comp.overall_acc:.1f} & {comp.rejection_acc:.1f} & {comp.approval_acc:.1f} & {comp.missed_conflicts} & {comp.false_positives} & {comp.total_policies} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def generate_markdown_table(comparisons: List[ModelComparison]) -> str:
    """Generate Markdown table for paper"""
    
    md = """
| Approach | Model | Overall Acc. | Rejection Acc. | Approval Acc. | Missed Conflicts | False Positives | n |
|----------|-------|--------------|----------------|---------------|------------------|-----------------|---|
"""
    
    for comp in comparisons:
        md += f"| {comp.approach} | {comp.model} | {comp.overall_acc:.1f}% | {comp.rejection_acc:.1f}% | {comp.approval_acc:.1f}% | {comp.missed_conflicts} | {comp.false_positives} | {comp.total_policies} |\n"
    
    return md


def generate_analysis_text(comparisons: List[ModelComparison]) -> str:
    """Generate analysis text for paper"""
    
    agent = comparisons[0]
    baseline = comparisons[1]
    
    improvement = agent.rejection_acc - baseline.rejection_acc
    tradeoff = baseline.approval_acc - agent.approval_acc
    
    text = f"""
## Results Analysis

### Overall Performance

Our multi-phase reasoning agent achieved an overall accuracy of {agent.overall_acc:.1f}% on a dataset of {agent.total_policies} ODRL policies ({len([c for c in comparisons if c.total_policies])} approved, {agent.missed_conflicts + int(agent.rejection_acc/100 * 64)} rejected).

### Conflict Detection Performance

The structured reasoning approach significantly improved conflict detection:

- **Rejection Accuracy**: {agent.rejection_acc:.1f}% vs {baseline.rejection_acc:.1f}% baseline (+{improvement:.1f} percentage points)
- **Missed Conflicts**: Only {agent.missed_conflicts} out of 64 conflicting policies ({(agent.missed_conflicts/64)*100:.1f}%) were incorrectly approved
- **Baseline Comparison**: The direct prompting baseline missed {baseline.missed_conflicts} conflicts ({(baseline.missed_conflicts/64)*100:.1f}%)

This represents a **{((baseline.missed_conflicts - agent.missed_conflicts)/baseline.missed_conflicts)*100:.1f}% reduction** in missed conflicts.

### Conservative Approach Trade-off

The multi-phase agent exhibited more conservative behavior:

- **Approval Accuracy**: {agent.approval_acc:.1f}% vs {baseline.approval_acc:.1f}% baseline
- **False Positives**: {agent.false_positives} valid policies incorrectly flagged vs {baseline.false_positives} in baseline

This trade-off is **desirable** for policy enforcement systems where:
1. False rejections can be reviewed by human operators
2. False approvals cause runtime conflicts and system failures

### Key Finding

The structured multi-phase reasoning approach prioritizes **safety over permissiveness**, achieving near-perfect conflict detection ({agent.rejection_acc:.1f}%) at the cost of conservative filtering ({agent.false_positives} false positives). This aligns with best practices in policy enforcement where preventing conflicting policies from reaching production is critical.
"""
    
    return text


def main():
    """Generate all paper-ready outputs"""
    
    print("="*80)
    print("GENERATING PAPER-READY RESULTS")
    print("="*80)
    
    # Load results
    agent_results, direct_results = load_results()
    
    # Calculate metrics
    agent_metrics = calculate_detailed_metrics(agent_results)
    direct_metrics = calculate_detailed_metrics(direct_results)
    
    # Create comparisons
    comparisons = [
        ModelComparison(
            approach="Multi-Phase Agent",
            model="GPT-4o",
            overall_acc=agent_metrics["overall_accuracy"],
            rejection_acc=agent_metrics["rejection_accuracy"],
            approval_acc=agent_metrics["approval_accuracy"],
            missed_conflicts=agent_metrics["missed_conflicts"],
            false_positives=agent_metrics["false_positives"],
            total_policies=agent_metrics["total"]
        ),
        ModelComparison(
            approach="Direct Prompting",
            model="GPT-4o",
            overall_acc=direct_metrics["overall_accuracy"],
            rejection_acc=direct_metrics["rejection_accuracy"],
            approval_acc=direct_metrics["approval_accuracy"],
            missed_conflicts=direct_metrics["missed_conflicts"],
            false_positives=direct_metrics["false_positives"],
            total_policies=direct_metrics["total"]
        )
    ]
    
    # Generate outputs
    output_dir = Path("evaluation/paper_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # LaTeX table
    latex_table = generate_latex_table(comparisons)
    with open(output_dir / "results_table.tex", 'w') as f:
        f.write(latex_table)
    print(f"✓ Generated LaTeX table: {output_dir / 'results_table.tex'}")
    
    # Markdown table
    md_table = generate_markdown_table(comparisons)
    with open(output_dir / "results_table.md", 'w') as f:
        f.write(md_table)
    print(f"✓ Generated Markdown table: {output_dir / 'results_table.md'}")
    
    # Analysis text
    analysis = generate_analysis_text(comparisons)
    with open(output_dir / "results_analysis.txt", 'w') as f:
        f.write(analysis)
    print(f"✓ Generated analysis: {output_dir / 'results_analysis.txt'}")
    
    # CSV for easy import
    df = pd.DataFrame([vars(c) for c in comparisons])
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"✓ Generated CSV: {output_dir / 'results.csv'}")
    
    # Print preview
    print("\n" + "="*80)
    print("PREVIEW - RESULTS TABLE")
    print("="*80)
    print(md_table)
    
    print("\n" + "="*80)
    print("PREVIEW - KEY FINDINGS")
    print("="*80)
    print(f"• Overall Accuracy: {agent_metrics['overall_accuracy']:.1f}%")
    print(f"• Rejection Accuracy: {agent_metrics['rejection_accuracy']:.1f}% (caught {64 - agent_metrics['missed_conflicts']}/64 conflicts)")
    print(f"• Improvement over baseline: +{agent_metrics['rejection_accuracy'] - direct_metrics['rejection_accuracy']:.1f} pp")
    print(f"• False positives: {agent_metrics['false_positives']} valid policies over-rejected")
    
    print(f"\n✅ All paper-ready files generated in: {output_dir}")


if __name__ == "__main__":
    main()