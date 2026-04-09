# evaluation/evaluate_reasoning_agent.py

"""
Evaluate YOUR Reasoning Agent on the unified dataset
Tests the 6-phase structured conflict detection
"""

import json
import argparse
import os
import importlib
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Type
from collections import Counter
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from evaluation.model_config_loader import load_model_config


def load_reasoner_class(module_path: str) -> Type:
    """Import Reasoner from a module path (e.g. agents.reasoner.reasoner_agent_old)."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, "Reasoner", None)
    if cls is None:
        raise ImportError(f"Module {module_path!r} has no attribute 'Reasoner'")
    return cls


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
    # Paper-style category (dataset ground_truth.conflict_primary + dataset_info mapping)
    gold_conflict_primary: Optional[str] = None
    predicted_paper_category: Optional[str] = None


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


# Order and labels aligned with rejected_policies_dataset.json / paper tables
PAPER_CATEGORY_ORDER = [
    "vagueness",
    "temporal",
    "spatial",
    "action_hierarchy",
    "role_hierarchy",
    "circular_dependency",
]

PAPER_CATEGORY_DISPLAY = {
    "vagueness": "Vagueness",
    "temporal": "Temporal",
    "spatial": "Spatial",
    "action_hierarchy": "Action Hierarchy",
    "role_hierarchy": "Role Hierarchy",
    "circular_dependency": "Circular Dependency",
}

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


def build_fine_type_to_paper_category(
    conflict_type_mapping: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Map normalized fine-grained conflict labels to dataset paper categories
    (keys: vagueness, temporal, spatial, action_hierarchy, role_hierarchy,
    circular_dependency).
    """
    out: Dict[str, str] = {}
    for paper_cat, fines in conflict_type_mapping.items():
        for f in fines:
            nf = normalize_conflict_type(f)
            if nf:
                out[nf] = paper_cat
    return out


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


def evaluate_policy(
    reasoner: Any,
    policy: dict,
    fine_to_paper: Optional[Dict[str, str]] = None,
) -> SimpleResult:
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
    gold_primary = policy.get("ground_truth", {}).get("conflict_primary")
    if isinstance(gold_primary, str):
        gold_primary = gold_primary.strip().lower() or None
    else:
        gold_primary = None
    
    try:
        # Call agent
        result = reasoner.reason(policy_text)
        raw_decision = str(result.get("decision", "needs_input")).lower()
        # Force binary decision: only "approve" is kept; all others become "reject".
        agent_decision = "approve" if raw_decision == "approve" else "reject"
        issues = result.get("issues", [])
        normalized_conflicts = [
            normalize_conflict_type(issue.get("conflict_type") or issue.get("category"))
            for issue in issues
        ]
        normalized_conflicts = [c for c in normalized_conflicts if c]
        # Keep only the first predicted conflict type, preserving model output order.
        predicted_conflicts = [normalized_conflicts[0]] if normalized_conflicts else []
        pred_paper_cat = None
        if fine_to_paper and predicted_conflicts:
            pred_paper_cat = fine_to_paper.get(predicted_conflicts[0])

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
            primary_conflict_match=primary_conflict_match,
            gold_conflict_primary=gold_primary,
            predicted_paper_category=pred_paper_cat,
        )
        
    except Exception as e:
        print(f"\n Error: {policy_id}: {str(e)[:100]}")
        return SimpleResult(
            policy_id=policy_id,
            expected=expected,
            expected_binary=expected_binary,
            # Keep binary behavior on exceptions as well.
            agent_decision="reject",
            correct=False,
            expected_conflicts=expected_conflicts,
            predicted_conflicts=[],
            primary_conflict_match=False if expected_binary == "REJECTED" and expected_conflicts else None,
            gold_conflict_primary=gold_primary,
            predicted_paper_category=None,
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


def write_agent_results_json_atomic(results_path: Path, results: List[SimpleResult]) -> None:
    """Persist the full results list after each policy; os.replace avoids half-written JSON on crash."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = results_path.parent / (results_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump([vars(r) for r in results], f, indent=2)
    os.replace(tmp_path, results_path)


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


def compute_paper_category_summary(
    results: List[SimpleResult],
) -> Optional[Dict[str, Any]]:
    """
    Category-specific conflict detection (paper-style): among REJECTED samples with a
    gold conflict_primary, fraction where the first predicted fine-grained type maps
    to the same paper category via dataset_info.conflict_type_mapping.

    Returns None when there are no qualifying rows (same condition as the printed table).
    """
    rows = [
        r
        for r in results
        if r.gold_conflict_primary and r.expected_binary == "REJECTED"
    ]
    if not rows:
        return None

    out_rows: List[Dict[str, Any]] = []
    total_n = 0
    total_ok = 0
    for key in PAPER_CATEGORY_ORDER:
        subset = [r for r in rows if r.gold_conflict_primary == key]
        n = len(subset)
        if n == 0:
            continue
        ok = sum(
            1
            for r in subset
            if r.predicted_paper_category == r.gold_conflict_primary
        )
        total_n += n
        total_ok += ok
        pct = (ok / n * 100) if n else 0.0
        out_rows.append(
            {
                "category_key": key,
                "category": PAPER_CATEGORY_DISPLAY.get(key, key),
                "n": n,
                "detected": ok,
                "detection_pct": round(pct, 1),
            }
        )

    overall = (total_ok / total_n * 100) if total_n else 0.0
    return {
        "rows": out_rows,
        "total_n": total_n,
        "total_detected": total_ok,
        "total_detection_pct": round(overall, 1),
    }


def paper_category_summary_to_latex(
    summary: Dict[str, Any],
    model_label: str,
    table_label: str = "tab:category_detection",
) -> str:
    """Booktabs-style table matching the paper's Category-Specific Conflict Detection layout."""
    lines: List[str] = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Category-Specific Conflict Detection ({model_label})}}",
        rf"\label{{{table_label}}}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Conflict Category} & \textbf{n} & \textbf{Detection (\%)} \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        name = str(row["category"]).replace("&", r"\&")
        lines.append(
            f"{name} & {int(row['n'])} & {float(row['detection_pct']):.1f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & \textbf{{{int(summary['total_n'])}}} & \textbf{{{float(summary['total_detection_pct']):.1f}}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def print_paper_category_table(results: List[SimpleResult]) -> None:
    summary = compute_paper_category_summary(results)
    if not summary:
        return

    print("\n" + "=" * 80)
    print("PAPER CATEGORY-SPECIFIC CONFLICT DETECTION (gold conflict_primary)")
    print("=" * 80)
    print(f"{'Category':<22} {'n':>6} {'Detection (%)':>14}")

    for row in summary["rows"]:
        label = row["category"]
        n = row["n"]
        pct = row["detection_pct"]
        print(f"{label:<22} {n:6d} {float(pct):14.1f}")

    print("-" * 44)
    print(
        f"{'Total':<22} {summary['total_n']:6d} {float(summary['total_detection_pct']):14.1f}"
    )


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
    parser.add_argument(
        "--approved-file",
        type=str,
        default=None,
        help="Path to approved policies JSON (default: data/approved_policies/approved_policies_dataset.json).",
    )
    parser.add_argument(
        "--rejected-file",
        type=str,
        default=None,
        help="Path to rejected policies JSON (default: data/rejected_policies/rejected_policies_dataset.json).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write per-policy results to this JSON path (default: evaluation/results/agent_results.json). "
             "File is rewritten after each policy (atomic replace).",
    )
    parser.add_argument(
        "--category-summary-json",
        type=str,
        default=None,
        help="Write paper-style category detection summary (JSON) to this path.",
    )
    parser.add_argument(
        "--category-latex",
        type=str,
        default=None,
        help="Write booktabs LaTeX table for category detection to this path.",
    )
    parser.add_argument(
        "--latex-caption-model",
        type=str,
        default=None,
        help="Model name for LaTeX \\caption{...} (default: resolved --model-id).",
    )
    parser.add_argument(
        "--latex-label",
        type=str,
        default="tab:category_detection",
        help="LaTeX \\label for --category-latex (default: tab:category_detection).",
    )
    parser.add_argument(
        "--reasoner-module",
        type=str,
        default="agents.reasoner.reasoner_agent",
        help="Python module that defines Reasoner (e.g. agents.reasoner.reasoner_agent_old).",
    )
    parser.add_argument(
        "--sample-approved",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N policies from the approved dataset (without replacement).",
    )
    parser.add_argument(
        "--sample-rejected",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N policies from the rejected dataset (without replacement).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for sampling (omit for nondeterministic samples).",
    )
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATING REASONING AGENT")
    print("="*80)
    
    # Load policies
    rejected_file = (
        Path(args.rejected_file)
        if args.rejected_file
        else Path("data/rejected_policies/rejected_policies_dataset.json")
    )
    approved_file = (
        Path(args.approved_file)
        if args.approved_file
        else Path("data/approved_policies/approved_policies_dataset.json")
    )
    
    rejected_policies: List[dict] = []
    approved_policies: List[dict] = []

    if rejected_file.exists():
        with open(rejected_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            rejected_policies = list(data["policies"])
            print(f"Loaded {len(rejected_policies)} REJECTED (file)")

    if approved_file.exists():
        with open(approved_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            approved_policies = list(data["policies"])
            print(f"Loaded {len(approved_policies)} APPROVED (file)")

    using_sample = (
        args.sample_approved is not None or args.sample_rejected is not None
    )
    if using_sample and (args.start != 0 or args.limit is not None):
        print(
            "Note: --start / --limit are ignored when "
            "--sample-approved or --sample-rejected is set."
        )

    if using_sample:
        rng = random.Random(args.seed) if args.seed is not None else random.Random()
        if args.sample_rejected is not None:
            want = args.sample_rejected
            have = len(rejected_policies)
            k = min(want, have)
            if k < want:
                print(
                    f"Warning: requested {want} rejected policies but only {have} in file; using {k}."
                )
            rejected_policies = (
                rng.sample(rejected_policies, k) if k else []
            )
            print(f"Sampled {len(rejected_policies)} REJECTED (requested {want})")
        if args.sample_approved is not None:
            want = args.sample_approved
            have = len(approved_policies)
            k = min(want, have)
            if k < want:
                print(
                    f"Warning: requested {want} approved policies but only {have} in file; using {k}."
                )
            approved_policies = (
                rng.sample(approved_policies, k) if k else []
            )
            print(f"Sampled {len(approved_policies)} APPROVED (requested {want})")
        # Same order as full-dataset path: rejected first, then approved.
        policies = rejected_policies + approved_policies
        full_total = len(rejected_policies) + len(approved_policies)
        print(f"\n Total after sampling: {full_total}")
        print(f" Evaluating:          {len(policies)}")
    else:
        policies = rejected_policies + approved_policies
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
            policies = policies[args.start : args.start + args.limit]

        print(f"\n Total loaded: {full_total}")
        print(f" Evaluating:   {len(policies)} (start={args.start}, limit={args.limit})")

    fine_to_paper: Dict[str, str] = {}
    if rejected_file.exists():
        with open(rejected_file, "r", encoding="utf-8") as f:
            rejected_meta = json.load(f)
        ctm = rejected_meta.get("dataset_info", {}).get("conflict_type_mapping", {})
        if isinstance(ctm, dict) and ctm:
            fine_to_paper = build_fine_type_to_paper_category(ctm)
    
    # Load model config
    model_config = load_model_config(args.model_id)
    print("\n Model config:")
    print(f"   model_id: {model_config['model_id']}")
    print(f"   base_url: {model_config['base_url']}")

    # Initialize agent
    Reasoner = load_reasoner_class(args.reasoner_module)
    print(f"\n Reasoner module: {args.reasoner_module}")
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

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.output_json) if args.output_json else (output_dir / "agent_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[SimpleResult] = []
    for i, policy in enumerate(policies, 1):
        print(f"{i}/{len(policies)}: {policy['policy_id'][:40]}...", end='\r')
        result = evaluate_policy(reasoner, policy, fine_to_paper=fine_to_paper)
        results.append(result)
        write_agent_results_json_atomic(results_path, results)

    if not policies:
        write_agent_results_json_atomic(results_path, results)

    print()

    # Metrics
    metrics = calculate_metrics(results)
    
    # Print
    print_results(metrics)
    print_paper_category_table(results)

    cat_summary = compute_paper_category_summary(results)
    if cat_summary and args.category_summary_json:
        p = Path(args.category_summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cat_summary, f, indent=2)
    if cat_summary and args.category_latex:
        cap = args.latex_caption_model or model_config["model_id"]
        tex = paper_category_summary_to_latex(
            cat_summary,
            model_label=cap,
            table_label=args.latex_label,
        )
        lp = Path(args.category_latex)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "w", encoding="utf-8") as f:
            f.write(tex)
    
    # Show missed
    missed = [
        r for r in results
        if r.expected_binary == "REJECTED" and r.agent_decision != "reject"
    ]
    if missed:
        print(f"\nMISSED {len(missed)} BAD POLICIES:")
        for r in missed[:5]:
            print(f"   - {r.policy_id}")
    
    print(f"\nDone! Saved to {results_path}")


if __name__ == "__main__":
    main()