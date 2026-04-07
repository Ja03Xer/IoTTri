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


def remove_if_exists(path: Path):
    if path.exists():
        path.unlink()


def get_criterion_score(evaluator_output, criterion_name):
    for item in evaluator_output["learner_evaluation"]["criteria"]:
        if item["criterion"] == criterion_name:
            return item["score"]
    return None


# ============================================================
# Setup
# ============================================================

def build_retriever():
    bootstrap_config = IoTTriConfig()
    retriever = IotSurfaceRetriever(
        str(PROJECT_ROOT / "knowledge_base" / "snippets.jsonl"),
        bootstrap_config.retrieval.embedding_model,
    )
    retriever.load_or_build()
    return retriever


def make_variant_config(
    *,
    memory_enabled: bool,
    memory_path: str,
    experiment_id: str,
) -> IoTTriConfig:
    config = IoTTriConfig()

    # keep adaptation fixed for both variants so only mistake memory changes
    config.adaptation.enabled = True

    # ablation switch
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
    }


# ============================================================
# Mistake-memory metrics
# ============================================================

def compute_repeated_error_rate(rows):
    """
    Repeated error rate:
    For consecutive cases within the same group, if both have accuracy <= 2,
    count that as a repeated error pair.

    Metric = repeated_error_pairs / total_related_pairs
    where total_related_pairs are consecutive same-group pairs with valid accuracy.
    """
    total_related_pairs = 0
    repeated_error_pairs = 0

    for i in range(len(rows) - 1):
        current_row = rows[i]
        next_row = rows[i + 1]

        if current_row["group"] != next_row["group"]:
            continue

        curr_acc = current_row["accuracy"]
        next_acc = next_row["accuracy"]

        if curr_acc is None or next_acc is None:
            continue

        total_related_pairs += 1

        if curr_acc <= 2 and next_acc <= 2:
            repeated_error_pairs += 1

    repeated_error_rate = round(
        100.0 * repeated_error_pairs / total_related_pairs, 4
    ) if total_related_pairs > 0 else 0.0

    return {
        "total_related_pairs": total_related_pairs,
        "repeated_error_pairs": repeated_error_pairs,
        "repeated_error_rate": repeated_error_rate,
    }


def compute_correction_speed(rows):
    """
    Correction speed:
    After a low-accuracy case (<=2), how many subsequent same-group cases
    does it take to reach accuracy >=3?

    Lower is better.
    If no correction occurs within the remaining same-group sequence, ignore that opportunity.

    We report:
    - average correction turns
    - corrected_within_1_rate
    """
    correction_turns = []
    total_error_opportunities = 0
    corrected_within_1 = 0

    # Build group index lists
    group_to_indices = {}
    for idx, row in enumerate(rows):
        group_to_indices.setdefault(row["group"], []).append(idx)

    for group, indices in group_to_indices.items():
        for pos, idx in enumerate(indices):
            acc = rows[idx]["accuracy"]
            if acc is None:
                continue
            if acc > 2:
                continue

            total_error_opportunities += 1

            found = False
            for next_pos in range(pos + 1, len(indices)):
                next_idx = indices[next_pos]
                next_acc = rows[next_idx]["accuracy"]
                if next_acc is None:
                    continue

                if next_acc >= 3:
                    turns_needed = next_pos - pos
                    correction_turns.append(turns_needed)
                    if turns_needed == 1:
                        corrected_within_1 += 1
                    found = True
                    break

            # If not found, we keep it as an error opportunity but no correction_turn entry

    avg_correction_turns = avg(correction_turns) if correction_turns else 0.0
    corrected_within_1_rate = round(
        100.0 * corrected_within_1 / total_error_opportunities, 4
    ) if total_error_opportunities > 0 else 0.0

    return {
        "error_opportunities": total_error_opportunities,
        "corrected_cases": len(correction_turns),
        "average_correction_turns": avg_correction_turns,
        "corrected_within_1_rate": corrected_within_1_rate,
    }


def build_run_summary(variant_rows):
    summary = {}

    for variant_name, rows in variant_rows.items():
        repeated_error = compute_repeated_error_rate(rows)
        correction_speed = compute_correction_speed(rows)

        summary[variant_name] = {
            "Accuracy": avg([r["accuracy"] for r in rows]),
            "Completeness": avg([r["completeness"] for r in rows]),
            "RepeatedError": repeated_error,
            "CorrectionSpeed": correction_speed,
        }

    return summary


# ============================================================
# Multi-run aggregation
# ============================================================

def aggregate_across_runs(run_summaries):
    def collect_metric(variant, metric):
        return [run_summary[variant][metric] for run_summary in run_summaries]

    def collect_repeated_error_metric(variant, metric):
        return [run_summary[variant]["RepeatedError"][metric] for run_summary in run_summaries]

    def collect_correction_metric(variant, metric):
        return [run_summary[variant]["CorrectionSpeed"][metric] for run_summary in run_summaries]

    aggregate = {}

    for variant in ["NoMemory", "WithMemory"]:
        aggregate[variant] = {
            "Accuracy_mean": avg(collect_metric(variant, "Accuracy")),
            "Accuracy_std": stddev(collect_metric(variant, "Accuracy")),
            "Completeness_mean": avg(collect_metric(variant, "Completeness")),
            "Completeness_std": stddev(collect_metric(variant, "Completeness")),
            "RepeatedError": {
                "total_related_pairs_mean": avg(collect_repeated_error_metric(variant, "total_related_pairs")),
                "repeated_error_pairs_mean": avg(collect_repeated_error_metric(variant, "repeated_error_pairs")),
                "repeated_error_rate_mean": avg(collect_repeated_error_metric(variant, "repeated_error_rate")),
                "repeated_error_rate_std": stddev(collect_repeated_error_metric(variant, "repeated_error_rate")),
            },
            "CorrectionSpeed": {
                "error_opportunities_mean": avg(collect_correction_metric(variant, "error_opportunities")),
                "corrected_cases_mean": avg(collect_correction_metric(variant, "corrected_cases")),
                "average_correction_turns_mean": avg(collect_correction_metric(variant, "average_correction_turns")),
                "average_correction_turns_std": stddev(collect_correction_metric(variant, "average_correction_turns")),
                "corrected_within_1_rate_mean": avg(collect_correction_metric(variant, "corrected_within_1_rate")),
                "corrected_within_1_rate_std": stddev(collect_correction_metric(variant, "corrected_within_1_rate")),
            },
        }

    return aggregate


# ============================================================
# Export
# ============================================================

def save_json(all_run_rows, all_run_summaries, aggregate, cold_start, num_runs):
    output_dir = PROJECT_ROOT / "examples" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "mistake_memory_ablation_results.json"
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

    path = output_dir / "mistake_memory_ablation_results.csv"
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
    print("| Variant | Repeated Error Rate ↓ | Avg Correction Turns ↓ | Corrected Within 1 Related Case ↑ |")
    print("|---|---:|---:|---:|")

    no_mem = run_summary["NoMemory"]
    with_mem = run_summary["WithMemory"]

    print(
        f"| No memory | "
        f"{no_mem['RepeatedError']['repeated_error_rate']:.2f}% | "
        f"{no_mem['CorrectionSpeed']['average_correction_turns']:.2f} | "
        f"{no_mem['CorrectionSpeed']['corrected_within_1_rate']:.2f}% |"
    )
    print(
        f"| With memory | "
        f"{with_mem['RepeatedError']['repeated_error_rate']:.2f}% | "
        f"{with_mem['CorrectionSpeed']['average_correction_turns']:.2f} | "
        f"{with_mem['CorrectionSpeed']['corrected_within_1_rate']:.2f}% |"
    )

    print("\n| Variant | Accuracy ↑ | Completeness ↑ |")
    print("|---|---:|---:|")
    print(
        f"| No memory | "
        f"{no_mem['Accuracy']:.2f} | "
        f"{no_mem['Completeness']:.2f} |"
    )
    print(
        f"| With memory | "
        f"{with_mem['Accuracy']:.2f} | "
        f"{with_mem['Completeness']:.2f} |"
    )


def print_main_table(aggregate):
    print("\n===== MISTAKE MEMORY ABLATION: FINAL AGGREGATED TABLE =====\n")
    print("| Variant | Repeated Error Rate ↓ | Avg Correction Turns ↓ | Corrected Within 1 Related Case ↑ |")
    print("|---|---:|---:|---:|")

    no_mem = aggregate["NoMemory"]
    with_mem = aggregate["WithMemory"]

    print(
        f"| No memory | "
        f"{no_mem['RepeatedError']['repeated_error_rate_mean']:.2f}% ± {no_mem['RepeatedError']['repeated_error_rate_std']:.2f} | "
        f"{no_mem['CorrectionSpeed']['average_correction_turns_mean']:.2f} ± {no_mem['CorrectionSpeed']['average_correction_turns_std']:.2f} | "
        f"{no_mem['CorrectionSpeed']['corrected_within_1_rate_mean']:.2f}% ± {no_mem['CorrectionSpeed']['corrected_within_1_rate_std']:.2f} |"
    )
    print(
        f"| With memory | "
        f"{with_mem['RepeatedError']['repeated_error_rate_mean']:.2f}% ± {with_mem['RepeatedError']['repeated_error_rate_std']:.2f} | "
        f"{with_mem['CorrectionSpeed']['average_correction_turns_mean']:.2f} ± {with_mem['CorrectionSpeed']['average_correction_turns_std']:.2f} | "
        f"{with_mem['CorrectionSpeed']['corrected_within_1_rate_mean']:.2f}% ± {with_mem['CorrectionSpeed']['corrected_within_1_rate_std']:.2f} |"
    )

    print("\n| Variant | Accuracy ↑ | Completeness ↑ |")
    print("|---|---:|---:|")
    print(
        f"| No memory | "
        f"{no_mem['Accuracy_mean']:.2f} ± {no_mem['Accuracy_std']:.2f} | "
        f"{no_mem['Completeness_mean']:.2f} ± {no_mem['Completeness_std']:.2f} |"
    )
    print(
        f"| With memory | "
        f"{with_mem['Accuracy_mean']:.2f} ± {with_mem['Accuracy_std']:.2f} | "
        f"{with_mem['Completeness_mean']:.2f} ± {with_mem['Completeness_std']:.2f} |"
    )


# ============================================================
# Main
# ============================================================

def main():
    num_runs = 10
    cold_start = True

    retriever = build_retriever()
    cases = build_test_cases()

    print(f"Running mistake-memory ablation on {len(cases)} test cases for {num_runs} repeated runs...\n")

    all_run_rows = []
    all_run_summaries = []

    for run_index in range(1, num_runs + 1):
        print(f"\n================ RUN {run_index}/{num_runs} ================\n")

        memory_path = PROJECT_ROOT / "examples" / "logs" / "mistake_memory_ablation_memory.jsonl"
        if cold_start:
            remove_if_exists(memory_path)

        variant_configs = {
            "NoMemory": make_variant_config(
                memory_enabled=False,
                memory_path=str(memory_path),
                experiment_id=f"mistake_memory_ablation_nomem_run_{run_index}",
            ),
            "WithMemory": make_variant_config(
                memory_enabled=True,
                memory_path=str(memory_path),
                experiment_id=f"mistake_memory_ablation_withmem_run_{run_index}",
            ),
        }

        variant_engines = {
            variant_name: IoTTriOrchestrator(
                config=variant_config,
                retriever=retriever,
                log_dir=str(PROJECT_ROOT / "examples" / "logs" / "mistake_memory_ablation" / f"run_{run_index}" / variant_name),
            )
            for variant_name, variant_config in variant_configs.items()
        }

        variant_rows = {
            "NoMemory": [],
            "WithMemory": [],
        }

        for case in cases:
            print(f"Running case: {case['case_id']} ({case['group']})")

            for variant_name in ["NoMemory", "WithMemory"]:
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

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()