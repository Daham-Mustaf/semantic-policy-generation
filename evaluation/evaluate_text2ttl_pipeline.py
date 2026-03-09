"""
End-to-End pipeline evaluation for text2ttl_GT.jsonl.

This version focuses on pipeline execution metrics only:
Reasoner -> Generator -> Validator.
No strict semantic alignment metrics are computed.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.reasoner.reasoner_agent import Reasoner
from agents.generator.generator import Generator
from agents.validator.validator_agent import ValidatorAgent
from evaluation.model_config_loader import load_model_config

try:
    from rdflib import Graph, Namespace, RDF
except Exception:
    Graph = None
    Namespace = None
    RDF = None


ODRL_NS = "http://www.w3.org/ns/odrl/2/"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
ODRL = Namespace(ODRL_NS) if Namespace else None

@dataclass
class SampleResult:
    sample_id: str
    input_text: str
    reasoner_decision: str
    reasoner_error: str
    generator_ran: bool
    generated_ok: bool
    generator_error: str
    validator_ran: bool
    validation_passed: bool
    validation_attempts: int
    validator_error: str
    generated_turtle: str
    final_turtle: str
    timings_ms: Dict[str, float]


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"\s+", " ", value).strip().lower()


def _norm_uri(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.startswith(ODRL_NS):
        return f"odrl:{text[len(ODRL_NS):]}"
    if text.startswith(XSD_NS):
        return f"xsd:{text[len(XSD_NS):]}"
    if "#" in text:
        return text.split("#")[-1]
    if "/" in text:
        return text.split("/")[-1]
    return text


def _norm_operand(value: Any) -> str:
    text = _norm_text(value)
    if not text:
        return ""
    datatype = ""
    if "^^" in text:
        text, datatype = text.split("^^", 1)
        text = text.strip()
        datatype = datatype.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    if datatype == "xsd:string":
        return re.sub(r"[\W_]+", "", text.lower())
    return text


def _norm_triplet(left: Any, op: Any, right: Any) -> Tuple[str, str, str]:
    return (_norm_text(left), _norm_text(op), _norm_operand(right))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _triplets_from_gt(value: Any) -> List[Tuple[str, str, str]]:
    triplets: List[Tuple[str, str, str]] = []
    for item in _as_list(value):
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            triplets.append(_norm_triplet(item[0], item[1], item[2]))
    return triplets


def extract_gold_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "policy_type": row.get("policy_type"),
        "permission_actions": row.get("Permission.actions", []),
        "permission_triplets": _triplets_from_gt(row.get("Permission.Constraints.Triplets", [])),
        "permission_duty_actions": row.get("Permission.duty.actions", row.get("Permission.duties", [])),
        "permission_duty_triplets": _triplets_from_gt(row.get("Permission.duty.Constraints.Triplets", [])),
        "prohibition_actions": row.get("Prohibition.actions", []),
        "prohibition_triplets": _triplets_from_gt(row.get("Prohibition.Constraints.Triplets", [])),
    }


def _literal_to_turtle_form(value: Any) -> str:
    if value is None:
        return ""
    if getattr(value, "datatype", None) is not None:
        return f"{value}^^{_norm_uri(value.datatype)}"
    if hasattr(value, "n3"):
        return _norm_uri(value)
    return str(value)


def _collect_rule_constraints(graph: Graph, rule_nodes: List[Any]) -> List[Tuple[str, str, str]]:
    triplets: List[Tuple[str, str, str]] = []
    for node in rule_nodes:
        for c in graph.objects(node, ODRL.constraint):
            left = None
            op = None
            right = None
            for v in graph.objects(c, ODRL.leftOperand):
                left = _norm_uri(v)
            for v in graph.objects(c, ODRL.operator):
                op = _norm_uri(v)
            for v in graph.objects(c, ODRL.rightOperand):
                right = _literal_to_turtle_form(v)
            for v in graph.objects(c, ODRL.rightOperandReference):
                right = _norm_uri(v)
            triplets.append(_norm_triplet(left, op, right))
    return triplets


def _collect_duty_actions_and_triplets(graph: Graph, rule_nodes: List[Any]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    actions: List[str] = []
    triplets: List[Tuple[str, str, str]] = []
    for node in rule_nodes:
        for duty in graph.objects(node, ODRL.duty):
            for action in graph.objects(duty, ODRL.action):
                actions.append(_norm_uri(action))
            for c in graph.objects(duty, ODRL.constraint):
                left = None
                op = None
                right = None
                for v in graph.objects(c, ODRL.leftOperand):
                    left = _norm_uri(v)
                for v in graph.objects(c, ODRL.operator):
                    op = _norm_uri(v)
                for v in graph.objects(c, ODRL.rightOperand):
                    right = _literal_to_turtle_form(v)
                for v in graph.objects(c, ODRL.rightOperandReference):
                    right = _norm_uri(v)
                triplets.append(_norm_triplet(left, op, right))
    return actions, triplets


def extract_from_turtle(turtle_str: str) -> Dict[str, Any]:
    if not turtle_str or Graph is None:
        return {}
    try:
        g = Graph()
        g.parse(data=turtle_str, format="turtle")
    except Exception:
        return {}

    policies = list(g.subjects(RDF.type, ODRL.Policy))
    if not policies:
        return {}
    policy = policies[0]

    policy_type = None
    for pt in g.objects(policy, RDF.type):
        if pt != ODRL.Policy:
            policy_type = _norm_uri(pt)
            break

    perm_nodes = list(g.objects(policy, ODRL.permission))
    proh_nodes = list(g.objects(policy, ODRL.prohibition))

    def collect_actions(nodes: List[Any]) -> List[str]:
        actions: List[str] = []
        for n in nodes:
            for a in g.objects(n, ODRL.action):
                actions.append(_norm_uri(a))
        return actions

    perm_duty_actions, perm_duty_triplets = _collect_duty_actions_and_triplets(g, perm_nodes)

    return {
        "policy_type": policy_type,
        "permission_actions": collect_actions(perm_nodes),
        "permission_triplets": _collect_rule_constraints(g, perm_nodes),
        "permission_duty_actions": perm_duty_actions,
        "permission_duty_triplets": perm_duty_triplets,
        "prohibition_actions": collect_actions(proh_nodes),
        "prohibition_triplets": _collect_rule_constraints(g, proh_nodes),
    }


def _to_set_list(values: Any) -> set:
    return {_norm_text(v) for v in _as_list(values) if _norm_text(v)}


def _to_set_triplets(values: Any) -> set:
    return {tuple(v) for v in _as_list(values) if isinstance(v, tuple) and any(v)}


def _prf(gold_set: set, pred_set: set) -> Tuple[float, float, float]:
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_structured_metrics(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalar_fields = ["policy_type"]
    list_fields = [
        "permission_actions",
        "permission_triplets",
        "permission_duty_actions",
        "permission_duty_triplets",
        "prohibition_actions",
        "prohibition_triplets",
    ]

    scalar_accuracy: Dict[str, float] = {}
    for field in scalar_fields:
        correct = 0
        for gold, pred in zip(gold_rows, pred_rows):
            if _norm_text(gold.get(field)) == _norm_text(pred.get(field)):
                correct += 1
        scalar_accuracy[field] = correct / len(gold_rows) if gold_rows else 0.0

    list_metrics: Dict[str, Dict[str, float]] = {}
    for field in list_fields:
        ps, rs, fs = [], [], []
        for gold, pred in zip(gold_rows, pred_rows):
            if "triplets" in field:
                gold_set = _to_set_triplets(gold.get(field, []))
                pred_set = _to_set_triplets(pred.get(field, []))
            else:
                gold_set = _to_set_list(gold.get(field, []))
                pred_set = _to_set_list(pred.get(field, []))
            p, r, f1 = _prf(gold_set, pred_set)
            ps.append(p)
            rs.append(r)
            fs.append(f1)
        list_metrics[field] = {
            "precision": sum(ps) / len(ps) if ps else 0.0,
            "recall": sum(rs) / len(rs) if rs else 0.0,
            "f1": sum(fs) / len(fs) if fs else 0.0,
        }

    e2e_correct = 0
    for gold, pred in zip(gold_rows, pred_rows):
        scalar_ok = all(_norm_text(gold.get(f)) == _norm_text(pred.get(f)) for f in scalar_fields)
        list_ok = True
        for f in list_fields:
            if "triplets" in f:
                if _to_set_triplets(gold.get(f, [])) != _to_set_triplets(pred.get(f, [])):
                    list_ok = False
                    break
            else:
                if _to_set_list(gold.get(f, [])) != _to_set_list(pred.get(f, [])):
                    list_ok = False
                    break
        if scalar_ok and list_ok:
            e2e_correct += 1

    return {
        "scalar_accuracy": scalar_accuracy,
        "list_metrics": list_metrics,
        "end_to_end_accuracy": (e2e_correct / len(gold_rows)) if gold_rows else 0.0,
    }


def load_text2ttl_jsonl(path: Path, dataset_size: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if dataset_size == -1:
        return rows
    if dataset_size < 1:
        raise ValueError("--dataset-size must be >= 1, or -1 for full dataset")
    return rows[:dataset_size]


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_single(
    reasoner: Reasoner,
    generator: Generator,
    validator: ValidatorAgent,
    row: Dict[str, Any],
    sample_id: str,
    respect_reasoner_gate: bool,
) -> SampleResult:
    text = row.get("Input", "")
    timings: Dict[str, float] = {}

    reasoner_decision = "ERROR"
    reasoner_error = ""
    generator_ran = False
    generated_ok = False
    generator_error = ""
    validator_ran = False
    validation_passed = False
    validation_attempts = 0
    validator_error = ""
    generated_turtle = ""
    final_turtle = ""

    t0 = time.perf_counter()
    try:
        reasoner_result = reasoner.reason(text)
        reasoner_decision = reasoner_result.get("decision", "needs_input").lower()
    except Exception as e:
        reasoner_error = str(e)
    timings["reasoner_ms"] = (time.perf_counter() - t0) * 1000

    if respect_reasoner_gate and reasoner_decision not in {"approve", "needs_input"}:
        return SampleResult(
            sample_id=sample_id,
            input_text=text,
            reasoner_decision=reasoner_decision,
            reasoner_error=reasoner_error,
            generator_ran=generator_ran,
            generated_ok=generated_ok,
            generator_error=generator_error,
            validator_ran=validator_ran,
            validation_passed=validation_passed,
            validation_attempts=validation_attempts,
            validator_error=validator_error,
            generated_turtle=generated_turtle,
            final_turtle=final_turtle,
            timings_ms=timings,
        )

    t1 = time.perf_counter()
    try:
        generator_ran = True
        gen_result = generator.generate(text, sample_id)
        generated_turtle = gen_result.get("odrl_turtle", "")
        final_turtle = generated_turtle
        generated_ok = bool(generated_turtle)
    except Exception as e:
        generator_error = str(e)
    timings["generator_ms"] = (time.perf_counter() - t1) * 1000

    if not generated_ok:
        return SampleResult(
            sample_id=sample_id,
            input_text=text,
            reasoner_decision=reasoner_decision,
            reasoner_error=reasoner_error,
            generator_ran=generator_ran,
            generated_ok=generated_ok,
            generator_error=generator_error,
            validator_ran=validator_ran,
            validation_passed=validation_passed,
            validation_attempts=validation_attempts,
            validator_error=validator_error,
            generated_turtle=generated_turtle,
            final_turtle=final_turtle,
            timings_ms=timings,
        )

    t2 = time.perf_counter()
    try:
        validator_ran = True
        vr = validator.validate_and_regenerate(
            policy_text=text,
            odrl_turtle=generated_turtle,
            max_attempts=3,
        )
        validation_passed = bool(vr.get("success", False))
        validation_attempts = int(vr.get("attempts", 0))
        final_turtle = vr.get("final_odrl", final_turtle)
    except Exception as e:
        validator_error = str(e)
    timings["validator_ms"] = (time.perf_counter() - t2) * 1000

    return SampleResult(
        sample_id=sample_id,
        input_text=text,
        reasoner_decision=reasoner_decision,
        reasoner_error=reasoner_error,
        generator_ran=generator_ran,
        generated_ok=generated_ok,
        generator_error=generator_error,
        validator_ran=validator_ran,
        validation_passed=validation_passed,
        validation_attempts=validation_attempts,
        validator_error=validator_error,
        generated_turtle=generated_turtle,
        final_turtle=final_turtle,
        timings_ms=timings,
    )


@dataclass
class PipelineMetrics:
    model_name: str
    total_policies: int
    reasoner_correct: int
    reasoner_accuracy: float
    correctly_approved: int
    incorrectly_rejected: int
    generator_attempts: int
    generated_success: int
    generation_success_rate: float
    validator_attempts: int
    first_attempt_valid: int
    final_valid: int
    validation_success_rate: float
    avg_regen_attempts: float
    end_to_end_success: int
    end_to_end_success_rate: float
    avg_reasoner_ms: float
    avg_generator_ms: float
    avg_validator_ms: float


def calculate_pipeline_metrics(model_name: str, results: List[SampleResult]) -> PipelineMetrics:
    total = len(results)
    reasoner_correct = sum(1 for r in results if r.reasoner_decision in {"approve", "needs_input"})
    correctly_approved = reasoner_correct
    incorrectly_rejected = sum(1 for r in results if r.reasoner_decision not in {"approve", "needs_input"})

    generator_attempts = sum(1 for r in results if r.generator_ran)
    generated_success = sum(1 for r in results if r.generated_ok)

    validator_attempts = sum(1 for r in results if r.validator_ran)
    first_attempt_valid = sum(1 for r in results if r.validation_passed and r.validation_attempts == 1)
    final_valid = sum(1 for r in results if r.validation_passed)

    needed_regen_attempts = [r.validation_attempts for r in results if r.validation_passed and r.validation_attempts > 1]
    avg_regen_attempts = _avg(needed_regen_attempts) if needed_regen_attempts else 0.0

    end_to_end_success = sum(
        1
        for r in results
        if r.reasoner_decision in {"approve", "needs_input"} and r.generated_ok and r.validation_passed
    )

    avg_reasoner_ms = _avg([r.timings_ms.get("reasoner_ms", 0.0) for r in results])
    avg_generator_ms = _avg([r.timings_ms.get("generator_ms", 0.0) for r in results if "generator_ms" in r.timings_ms])
    avg_validator_ms = _avg([r.timings_ms.get("validator_ms", 0.0) for r in results if "validator_ms" in r.timings_ms])

    return PipelineMetrics(
        model_name=model_name,
        total_policies=total,
        reasoner_correct=reasoner_correct,
        reasoner_accuracy=(reasoner_correct / total * 100.0) if total else 0.0,
        correctly_approved=correctly_approved,
        incorrectly_rejected=incorrectly_rejected,
        generator_attempts=generator_attempts,
        generated_success=generated_success,
        generation_success_rate=(generated_success / generator_attempts * 100.0) if generator_attempts else 0.0,
        validator_attempts=validator_attempts,
        first_attempt_valid=first_attempt_valid,
        final_valid=final_valid,
        validation_success_rate=(final_valid / validator_attempts * 100.0) if validator_attempts else 0.0,
        avg_regen_attempts=avg_regen_attempts,
        end_to_end_success=end_to_end_success,
        end_to_end_success_rate=(end_to_end_success / total * 100.0) if total else 0.0,
        avg_reasoner_ms=avg_reasoner_ms,
        avg_generator_ms=avg_generator_ms,
        avg_validator_ms=avg_validator_ms,
    )


def print_pipeline_results(metrics: PipelineMetrics) -> None:
    print("\n" + "=" * 100)
    print(f"🔄 END-TO-END PIPELINE RESULTS - {metrics.model_name}")
    print("=" * 100)

    print("\n📊 OVERALL METRICS")
    print("-" * 100)
    print(f"Total Policies Evaluated:  {metrics.total_policies}")
    print(f"Reasoner Accuracy:         {metrics.reasoner_accuracy:.1f}%")
    print(f"End-to-End Success Rate:   {metrics.end_to_end_success_rate:.1f}%")

    print("\n" + "=" * 100)
    print("🤔 STAGE 1: REASONER (Conflict Detection)")
    print("=" * 100)
    print(f"Correct Decisions:         {metrics.reasoner_correct}/{metrics.total_policies}")
    print(f"Correctly Approved:        {metrics.correctly_approved}")
    print(f"Incorrectly Rejected:      {metrics.incorrectly_rejected}")
    print(f"Accuracy:                  {metrics.reasoner_accuracy:.1f}%")

    print("\n" + "=" * 100)
    print("⚙️  STAGE 2: GENERATOR (ODRL Creation)")
    print("=" * 100)
    print(f"Generation Attempts:       {metrics.generator_attempts}")
    print(f"Successfully Generated:    {metrics.generated_success}")
    print(f"Success Rate:              {metrics.generation_success_rate:.1f}%")

    print("\n" + "=" * 100)
    print("✅ STAGE 3: VALIDATOR (SHACL Conformance)")
    print("=" * 100)
    print(f"Validation Attempts:       {metrics.validator_attempts}")
    print(f"Valid on First Attempt:    {metrics.first_attempt_valid}")
    print(f"Valid After Regeneration:  {metrics.final_valid - metrics.first_attempt_valid}")
    print(f"Final Valid:               {metrics.final_valid}")
    print(f"Validation Success Rate:   {metrics.validation_success_rate:.1f}%")
    print(f"Avg Regen Attempts:        {metrics.avg_regen_attempts:.2f}")


def build_console_metrics_view(
    metrics: PipelineMetrics,
    generation_structured: Dict[str, Any],
    final_structured: Dict[str, Any],
) -> Dict[str, Any]:
    """Build nested metrics view for easier terminal reading."""
    first_attempt_valid_rate = (
        metrics.first_attempt_valid / metrics.validator_attempts * 100.0
        if metrics.validator_attempts
        else 0.0
    )
    return {
        "structured_metrics": {
            "generation_output": generation_structured,
            "final_output_after_validation": final_structured,
            "reasoner": {
                "correct_decisions": metrics.reasoner_correct,
                "total": metrics.total_policies,
                "accuracy": round(metrics.reasoner_accuracy, 3),
            },
            "end_to_end": {
                "success": metrics.end_to_end_success,
                "total": metrics.total_policies,
                "success_rate": round(metrics.end_to_end_success_rate, 3),
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline evaluation on text2ttl_GT.jsonl.")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--dataset-size", type=int, default=5)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/text2policy/text2ttl_GT.jsonl",
    )
    parser.add_argument(
        "--respect-reasoner-gate",
        action="store_true",
        help="If set, skip generator/validator when reasoner decision is not approve.",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("🚀 END-TO-END PIPELINE EVALUATION")
    print("=" * 100)

    model_config = load_model_config(args.model_id)
    llm_cfg = {
        "api_key": model_config["api_key"],
        "base_url": model_config["base_url"],
        "model": model_config["model_id"],
    }
    model_name = model_config["model_id"]

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    rows = load_text2ttl_jsonl(dataset_path, args.dataset_size)

    print("\n🔧 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Base URL: {model_config['base_url']}")
    if args.dataset_size == -1:
        print(f"   ✓ Loaded {len(rows)} text2ttl samples (full dataset)")
    else:
        print(f"   ✓ Loaded {len(rows)} text2ttl samples (first {args.dataset_size} items)")
    print(f"   Reasoner gate respected: {args.respect_reasoner_gate}")

    print("\n🤖 Initializing pipeline agents...")
    reasoner = Reasoner(**llm_cfg, temperature=0.0)
    generator = Generator(**llm_cfg, temperature=0.0)
    validator = ValidatorAgent(**llm_cfg, temperature=0.0)
    print("   ✓ Reasoner initialized")
    print("   ✓ Generator initialized")
    print("   ✓ Validator initialized")

    sample_results: List[SampleResult] = []
    gold_rows: List[Dict[str, Any]] = []
    generation_pred_rows: List[Dict[str, Any]] = []
    final_pred_rows: List[Dict[str, Any]] = []
    print(f"\n{'=' * 100}")
    print("⚙️  RUNNING PIPELINE EVALUATION...")
    print("=" * 100)

    for idx, row in enumerate(rows, start=1):
        sample_id = f"text2ttl_{idx:03d}"
        print(f"\n[{idx}/{len(rows)}] Processing: {sample_id}")
        print("-" * 100)

        result = evaluate_single(
            reasoner=reasoner,
            generator=generator,
            validator=validator,
            row=row,
            sample_id=sample_id,
            respect_reasoner_gate=args.respect_reasoner_gate,
        )
        sample_results.append(result)
        gold_rows.append(extract_gold_fields(row))
        generation_pred_rows.append(extract_from_turtle(result.generated_turtle))
        final_pred_rows.append(extract_from_turtle(result.final_turtle))

        print(
            f"   Reasoner: {result.reasoner_decision.upper():<10} "
            f"({'✓' if result.reasoner_decision in {'approve', 'needs_input'} else '✗'})"
        )
        if result.generator_ran:
            print(f"   Generator: {'SUCCESS' if result.generated_ok else 'FAILED':<10}")
        if result.validator_ran:
            print(f"   Validator: {'VALID' if result.validation_passed else 'INVALID':<10} (attempts: {result.validation_attempts})")
        pipeline_ok = (
            result.reasoner_decision in {"approve", "needs_input"}
            and result.generated_ok
            and result.validation_passed
        )
        print(f"   Pipeline: {'SUCCESS' if pipeline_ok else '❌ FAILED'}")

    metrics = calculate_pipeline_metrics(model_name, sample_results)
    generation_structured = evaluate_structured_metrics(gold_rows, generation_pred_rows)
    final_structured = evaluate_structured_metrics(gold_rows, final_pred_rows)

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace(":", "_").replace("(", "").replace(")", "")
    base_name = f"{safe_name}_text2ttl_pipeline"

    details_path = output_dir / f"{base_name}_details.json"
    metrics_path = output_dir / f"{base_name}_metrics.json"

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": [asdict(r) for r in sample_results],
                "gold_rows": gold_rows,
                "generation_pred_rows": generation_pred_rows,
                "final_pred_rows": final_pred_rows,
                "dataset_path": str(dataset_path),
                "total_samples": len(sample_results),
                "respect_reasoner_gate": args.respect_reasoner_gate,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(metrics),
                "structured_metrics": {
                    "generation_output": generation_structured,
                    "final_output_after_validation": final_structured,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print_pipeline_results(metrics)
    print("\n📦 METRICS (JSON VIEW)")
    print("-" * 100)
    print(
        json.dumps(
            build_console_metrics_view(metrics, generation_structured, final_structured),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n✅ Complete! Results saved to:")
    print(f"   • {details_path}")
    print(f"   • {metrics_path}")


if __name__ == "__main__":
    main()
