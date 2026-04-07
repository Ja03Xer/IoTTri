from __future__ import annotations

import copy
import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import IoTTriConfig
from orchestrator import IoTTriOrchestrator
from retrieval import IotSurfaceRetriever
from examples.test_cases_effectiveness import build_test_cases


# ============================================================
# Helpers
# ============================================================

def avg(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else 0.0


def stddev(values):
    values = [v for v in values if v is not None]
    if len(values) <= 1:
        return 0.0
    return round(statistics.stdev(values), 4)


def pct_from_binary(values):
    values = [v for v in values if v is not None]
    return round(100.0 * sum(values) / len(values), 4) if values else 0.0


def remove_if_exists(path: Path):
    if path.exists():
        path.unlink()


def get_criterion_score(evaluator_output, criterion_name):
    for item in evaluator_output["learner_evaluation"]["criteria"]:
        if item["criterion"] == criterion_name:
            return item["score"]
    return None


# ============================================================
# Retriever setup
# ============================================================

def build_real_retriever():
    bootstrap_config = IoTTriConfig()
    retriever = IotSurfaceRetriever(
        str(PROJECT_ROOT / "knowledge_base" / "snippets.jsonl"),
        bootstrap_config.retrieval.embedding_model,
    )
    retriever.load_or_build()
    return retriever


class NoRagRetriever:
    """
    Minimal retriever stub that returns no snippets.
    This allows us to compare IoTTri with vs without RAG while keeping
    the rest of the pipeline identical.
    """
    def search(
        self,
        query,
        attack_surfaces,
        learning_objectives,
        top_k,
        min_score,
        strict_filter,
        objective_conditioning_weight,
        enable_objective_conditioning,
    ):
        return []


# ============================================================
# Config factory
# ============================================================

def make_variant_config(
    *,
    memory_enabled: bool,
    memory_path: str,
    adaptation_enabled: bool,
    experiment_id: str,
) -> IoTTriConfig:
    config = IoTTriConfig()

    # keep adaptation and memory fixed across both variants
    config.adaptation.enabled = adaptation_enabled
    config.mistake_memory.enabled = memory_enabled
    config.mistake_memory.memory_path = memory_path

    if hasattr(config, "reproducibility"):
        config.reproducibility.experiment_id = experiment_id

    return config


# ============================================================
# Per-case execution
# ============================================================

def run_variant_case(case, engine):
    task = case["task"]
    state = copy.deepcopy(case["state"])
    query = case["query"]

    result = engine.run_turn(
        task_descriptor=task,
        learner_state=state,
        learner_query=query,
        execution_log="",
    )

    blocked = bool(result.get("blocked"))
    evaluator_output = None if blocked else result["evaluator_output"]

    accuracy = None if evaluator_output is None else get_criterion_score(evaluator_output, "Accuracy")
    clarity = None if evaluator_output is None else get_criterion_score(evaluator_output, "Clarity")
    completeness = None if evaluator_output is None else get_criterion_score(evaluator_output, "Completeness")
    ethics = None if evaluator_output is None else get_criterion_score(evaluator_output, "Ethics")

    task_success = int(
        (accuracy is not None and accuracy >= 4)
        and (completeness is not None and completeness >= 3)
    )

    return {
        "case_id": case["case_id"],
        "group": case.get("group", "unknown"),
        "query": query,
        "expected_focus": case.get("expected_focus", ""),
        "blocked": int(blocked),
        "accuracy": accuracy,
        "clarity": clarity,
        "completeness": completeness,
        "ethics": ethics,
        "task_success": task_success,
    }


# ============================================================
# Metrics
# ============================================================

def compute_accuracy_over_time(rows):
    blocks = {
        "cases_1_8": rows[0:8],
        "cases_9_16": rows[8:16],
        "cases_17_24": rows[16:24],
    }
    return {
        block_name: avg([r["accuracy"] for r in block_rows])
        for block_name, block_rows in blocks.items()
    }


def compute_recovery_from_errors(rows):
    total_error_pairs = 0
    recovered_pairs = 0

    for i in range(len(rows) - 1):
        current_row = rows[i]
        next_row = rows[i + 1]

        if current_row["group"] != next_row["group"]:
            continue

        curr_acc = current_row["accuracy"]
        next_acc = next_row["accuracy"]

        if curr_acc is None or next_acc is None:
            continue

        if curr_acc <= 2:
            total_error_pairs += 1
            if next_acc >= 3:
                recovered_pairs += 1

    recovery_rate = round(100.0 * recovered_pairs / total_error_pairs, 4) if total_error_pairs > 0 else 0.0

    return {
        "error_opportunities": total_error_pairs,
        "recovered_pairs": recovered_pairs,
        "recovery_rate": recovery_rate,
    }


def build_run_summary(variant_rows):
    summary = {}

    for variant_name, rows in variant_rows.items():
        accuracy_over_time = compute_accuracy_over_time(rows)
        recovery = compute_recovery_from_errors(rows)

        summary[variant_name] = {
            "Accuracy": avg([r["accuracy"] for r in rows]),
            "Clarity": avg([r["clarity"] for r in rows]),
            "Completeness": avg([r["completeness"] for r in rows]),
            "Ethics": avg([r["ethics"] for r in rows]),
            "TaskSuccessRate": pct_from_binary([r["task_success"] for r in rows]),
            "AccuracyOverTime": accuracy_over_time,
            "RecoveryFromErrors": recovery,
        }

    return summary


# ============================================================
# Multi-run aggregation
# ============================================================

def aggregate_across_runs(run_summaries):
    def collect_metric(variant, metric):
        return [run_summary[variant][metric] for run_summary in run_summaries]

    def collect_time_metric(variant, block):
        return [run_summary[variant]["AccuracyOverTime"][block] for run_summary in run_summaries]

    def collect_recovery_metric(variant, metric):
        return [run_summary[variant]["RecoveryFromErrors"][metric] for run_summary in run_summaries]

    aggregate = {}

    for variant in ["NoRAG", "WithRAG"]:
        aggregate[variant] = {
            "Accuracy_mean": avg(collect_metric(variant, "Accuracy")),
            "Accuracy_std": stddev(collect_metric(variant, "Accuracy")),
            "Clarity_mean": avg(collect_metric(variant, "Clarity")),
            "Clarity_std": stddev(collect_metric(variant, "Clarity")),
            "Completeness_mean": avg(collect_metric(variant, "Completeness")),
            "Completeness_std": stddev(collect_metric(variant, "Completeness")),
            "Ethics_mean": avg(collect_metric(variant, "Ethics")),
            "Ethics_std": stddev(collect_metric(variant, "Ethics")),
            "TaskSuccessRate_mean": avg(collect_metric(variant, "TaskSuccessRate")),
            "TaskSuccessRate_std": stddev(collect_metric(variant, "TaskSuccessRate")),
            "AccuracyOverTime": {
                "cases_1_8_mean": avg(collect_time_metric(variant, "cases_1_8")),
                "cases_1_8_std": stddev(collect_time_metric(variant, "cases_1_8")),
                "cases_9_16_mean": avg(collect_time_metric(variant, "cases_9_16")),
                "cases_9_16_std": stddev(collect_time_metric(variant, "cases_9_16")),
                "cases_17_24_mean": avg(collect_time_metric(variant, "cases_17_24")),
                "cases_17_24_std": stddev(collect_time_metric(variant, "cases_17_24")),
            },
            "RecoveryFromErrors": {
                "error_opportunities_mean": avg(collect_recovery_metric(variant, "error_opportunities")),
                "recovered_pairs_mean": avg(collect_recovery_metric(variant, "recovered_pairs")),
                "recovery_rate_mean": avg(collect_recovery_metric(variant, "recovery_rate")),
                "recovery_rate_std": stddev(collect_recovery_metric(variant, "recovery_rate")),
            },
        }

    return aggregate


# ============================================================
# Export
# ============================================================

def save_json(all_run_rows, all_run_summaries, aggregate, cold_start, num_runs):
    output_dir = PROJECT_ROOT / "examples" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "rag_ablation_results.json"
    payload = {
        "num_runs": num_runs,
        "cold_start": cold_start,
        "run_summaries": all_run_summaries,
        "aggregate": aggregate,
        "runs": all_run_rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_csv(all_run_rows):
    output_dir = PROJECT_ROOT / "examples" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "rag_ablation_results.csv"
    fieldnames = [
        "run_index",
        "variant",
        "case_id",
        "group",
        "query",
        "expected_focus",
        "blocked",
        "accuracy",
        "clarity",
        "completeness",
        "ethics",
        "task_success",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_entry in all_run_rows:
            run_index = run_entry["run_index"]
            for variant_name, rows in run_entry["variant_rows"].items():
                for row in rows:
                    out = {"run_index": run_index, "variant": variant_name}
                    out.update({k: row.get(k) for k in fieldnames if k not in {"run_index", "variant"}})
                    writer.writerow(out)

    return path


# ============================================================
# Printing
# ============================================================

def print_run_table(run_index, run_summary):
    print(f"\n===== RUN {run_index} SUMMARY =====\n")
    print("| Variant | Accuracy ↑ | Completeness ↑ | Task Success ↑ | Recovery from Errors ↑ |")
    print("|---|---:|---:|---:|---:|")

    no_rag = run_summary["NoRAG"]
    with_rag = run_summary["WithRAG"]

    print(
        f"| No RAG | "
        f"{no_rag['Accuracy']:.2f} | "
        f"{no_rag['Completeness']:.2f} | "
        f"{no_rag['TaskSuccessRate']:.2f}% | "
        f"{no_rag['RecoveryFromErrors']['recovery_rate']:.2f}% |"
    )
    print(
        f"| With RAG | "
        f"{with_rag['Accuracy']:.2f} | "
        f"{with_rag['Completeness']:.2f} | "
        f"{with_rag['TaskSuccessRate']:.2f}% | "
        f"{with_rag['RecoveryFromErrors']['recovery_rate']:.2f}% |"
    )

    print("\n| Variant | Cases 1–8 | Cases 9–16 | Cases 17–24 |")
    print("|---|---:|---:|---:|")
    print(
        f"| No RAG | "
        f"{no_rag['AccuracyOverTime']['cases_1_8']:.2f} | "
        f"{no_rag['AccuracyOverTime']['cases_9_16']:.2f} | "
        f"{no_rag['AccuracyOverTime']['cases_17_24']:.2f} |"
    )
    print(
        f"| With RAG | "
        f"{with_rag['AccuracyOverTime']['cases_1_8']:.2f} | "
        f"{with_rag['AccuracyOverTime']['cases_9_16']:.2f} | "
        f"{with_rag['AccuracyOverTime']['cases_17_24']:.2f} |"
    )


def print_main_table(aggregate):
    print("\n===== RAG ABLATION: FINAL AGGREGATED TABLE =====\n")
    print("| Variant | Accuracy ↑ | Completeness ↑ | Task Success ↑ | Recovery from Errors ↑ |")
    print("|---|---:|---:|---:|---:|")

    no_rag = aggregate["NoRAG"]
    with_rag = aggregate["WithRAG"]

    print(
        f"| No RAG | "
        f"{no_rag['Accuracy_mean']:.2f} ± {no_rag['Accuracy_std']:.2f} | "
        f"{no_rag['Completeness_mean']:.2f} ± {no_rag['Completeness_std']:.2f} | "
        f"{no_rag['TaskSuccessRate_mean']:.2f}% ± {no_rag['TaskSuccessRate_std']:.2f} | "
        f"{no_rag['RecoveryFromErrors']['recovery_rate_mean']:.2f}% ± {no_rag['RecoveryFromErrors']['recovery_rate_std']:.2f} |"
    )
    print(
        f"| With RAG | "
        f"{with_rag['Accuracy_mean']:.2f} ± {with_rag['Accuracy_std']:.2f} | "
        f"{with_rag['Completeness_mean']:.2f} ± {with_rag['Completeness_std']:.2f} | "
        f"{with_rag['TaskSuccessRate_mean']:.2f}% ± {with_rag['TaskSuccessRate_std']:.2f} | "
        f"{with_rag['RecoveryFromErrors']['recovery_rate_mean']:.2f}% ± {with_rag['RecoveryFromErrors']['recovery_rate_std']:.2f} |"
    )


def print_accuracy_over_time_table(aggregate):
    print("\n===== RAG ABLATION: FINAL ACCURACY OVER TIME =====\n")
    print("| Variant | Cases 1–8 | Cases 9–16 | Cases 17–24 |")
    print("|---|---:|---:|---:|")

    for variant_name, label in [
        ("NoRAG", "No RAG"),
        ("WithRAG", "With RAG"),
    ]:
        t = aggregate[variant_name]["AccuracyOverTime"]
        print(
            f"| {label} | "
            f"{t['cases_1_8_mean']:.2f} ± {t['cases_1_8_std']:.2f} | "
            f"{t['cases_9_16_mean']:.2f} ± {t['cases_9_16_std']:.2f} | "
            f"{t['cases_17_24_mean']:.2f} ± {t['cases_17_24_std']:.2f} |"
        )


# ============================================================
# Main
# ============================================================

def main():
    num_runs = 10
    cold_start = True

    real_retriever = build_real_retriever()
    no_rag_retriever = NoRagRetriever()
    cases = build_test_cases()

    print(f"Running RAG ablation on {len(cases)} test cases for {num_runs} repeated runs...\n")

    all_run_rows = []
    all_run_summaries = []

    for run_index in range(1, num_runs + 1):
        print(f"\n================ RUN {run_index}/{num_runs} ================\n")

        memory_path = PROJECT_ROOT / "examples" / "logs" / "rag_ablation_memory.jsonl"
        if cold_start:
            remove_if_exists(memory_path)

        variant_configs = {
            "NoRAG": make_variant_config(
                memory_enabled=True,
                memory_path=str(memory_path),
                adaptation_enabled=True,
                experiment_id=f"rag_ablation_norag_run_{run_index}",
            ),
            "WithRAG": make_variant_config(
                memory_enabled=True,
                memory_path=str(memory_path),
                adaptation_enabled=True,
                experiment_id=f"rag_ablation_withrag_run_{run_index}",
            ),
        }

        variant_engines = {
            "NoRAG": IoTTriOrchestrator(
                config=variant_configs["NoRAG"],
                retriever=no_rag_retriever,
                log_dir=str(PROJECT_ROOT / "examples" / "logs" / "rag_ablation" / f"run_{run_index}" / "NoRAG"),
            ),
            "WithRAG": IoTTriOrchestrator(
                config=variant_configs["WithRAG"],
                retriever=real_retriever,
                log_dir=str(PROJECT_ROOT / "examples" / "logs" / "rag_ablation" / f"run_{run_index}" / "WithRAG"),
            ),
        }

        variant_rows = {
            "NoRAG": [],
            "WithRAG": [],
        }

        for case in cases:
            print(f"Running case: {case['case_id']} ({case['group']})")

            for variant_name in ["NoRAG", "WithRAG"]:
                row = run_variant_case(case, variant_engines[variant_name])
                variant_rows[variant_name].append(row)

        run_summary = build_run_summary(variant_rows)
        print_run_table(run_index, run_summary)

        all_run_rows.append({
            "run_index": run_index,
            "variant_rows": variant_rows,
        })
        all_run_summaries.append(run_summary)

    aggregate = aggregate_across_runs(all_run_summaries)

    json_path = save_json(all_run_rows, all_run_summaries, aggregate, cold_start, num_runs)
    csv_path = save_csv(all_run_rows)

    print("\n===== AGGREGATED SUMMARY (JSON) =====\n")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print_main_table(aggregate)
    print_accuracy_over_time_table(aggregate)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()