from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = PROJECT_ROOT / "examples" / "baseline_comparison.py"
LOG_DIR = PROJECT_ROOT / "examples" / "logs"
SUMMARY_JSON = LOG_DIR / "effectiveness_vs_baseline_results.json"
MISTAKE_MEMORY = LOG_DIR / "mistake_memory.jsonl"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stddev(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def remove_if_exists(path: Path) -> None:
    if path.exists():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def run_once(run_index: int, cold_start: bool = True) -> dict:
    print(f"\n================ RUN {run_index} ================\n")

    if cold_start:
        remove_if_exists(MISTAKE_MEMORY)

    remove_if_exists(SUMMARY_JSON)

    result = subprocess.run(
        [sys.executable, str(BASELINE_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Run {run_index} failed with exit code {result.returncode}")

    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(f"Expected summary JSON not found: {SUMMARY_JSON}")

    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    summary = data["summary"]

    return {
        "run_index": run_index,
        "summary": summary,
    }


def aggregate_runs(run_results: list[dict]) -> dict:
    iottri_acc = [r["summary"]["IoTTri"]["Accuracy"] for r in run_results]
    iottri_cla = [r["summary"]["IoTTri"]["Clarity"] for r in run_results]
    iottri_com = [r["summary"]["IoTTri"]["Completeness"] for r in run_results]
    iottri_eth = [r["summary"]["IoTTri"]["Ethics"] for r in run_results]

    base_acc = [r["summary"]["SingleLLMBaseline"]["Accuracy"] for r in run_results]
    base_cla = [r["summary"]["SingleLLMBaseline"]["Clarity"] for r in run_results]
    base_com = [r["summary"]["SingleLLMBaseline"]["Completeness"] for r in run_results]
    base_eth = [r["summary"]["SingleLLMBaseline"]["Ethics"] for r in run_results]

    return {
        "num_runs": len(run_results),
        "IoTTri": {
            "Accuracy_mean": round(mean(iottri_acc), 3),
            "Accuracy_std": round(stddev(iottri_acc), 3),
            "Clarity_mean": round(mean(iottri_cla), 3),
            "Clarity_std": round(stddev(iottri_cla), 3),
            "Completeness_mean": round(mean(iottri_com), 3),
            "Completeness_std": round(stddev(iottri_com), 3),
            "Ethics_mean": round(mean(iottri_eth), 3),
            "Ethics_std": round(stddev(iottri_eth), 3),
        },
        "SingleLLMBaseline": {
            "Accuracy_mean": round(mean(base_acc), 3),
            "Accuracy_std": round(stddev(base_acc), 3),
            "Clarity_mean": round(mean(base_cla), 3),
            "Clarity_std": round(stddev(base_cla), 3),
            "Completeness_mean": round(mean(base_com), 3),
            "Completeness_std": round(stddev(base_com), 3),
            "Ethics_mean": round(mean(base_eth), 3),
            "Ethics_std": round(stddev(base_eth), 3),
        },
    }


def print_table(aggregate: dict) -> None:
    print("\n===== REPEATED-RUN SUMMARY TABLE =====\n")
    print("| System | Accuracy | Clarity | Completeness | Ethics |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| IoTTri | "
        f"{aggregate['IoTTri']['Accuracy_mean']:.3f} ± {aggregate['IoTTri']['Accuracy_std']:.3f} | "
        f"{aggregate['IoTTri']['Clarity_mean']:.3f} ± {aggregate['IoTTri']['Clarity_std']:.3f} | "
        f"{aggregate['IoTTri']['Completeness_mean']:.3f} ± {aggregate['IoTTri']['Completeness_std']:.3f} | "
        f"{aggregate['IoTTri']['Ethics_mean']:.3f} ± {aggregate['IoTTri']['Ethics_std']:.3f} |"
    )
    print(
        f"| Single-LLM Baseline | "
        f"{aggregate['SingleLLMBaseline']['Accuracy_mean']:.3f} ± {aggregate['SingleLLMBaseline']['Accuracy_std']:.3f} | "
        f"{aggregate['SingleLLMBaseline']['Clarity_mean']:.3f} ± {aggregate['SingleLLMBaseline']['Clarity_std']:.3f} | "
        f"{aggregate['SingleLLMBaseline']['Completeness_mean']:.3f} ± {aggregate['SingleLLMBaseline']['Completeness_std']:.3f} | "
        f"{aggregate['SingleLLMBaseline']['Ethics_mean']:.3f} ± {aggregate['SingleLLMBaseline']['Ethics_std']:.3f} |"
    )


def main():
    # Change this to 10, 12, or 15 as you prefer
    num_runs = 2

    # cold_start=True => delete mistake memory before each run
    # cold_start=False => keep persistent memory across runs
    cold_start = True

    run_results = []
    for i in range(1, num_runs + 1):
        run_results.append(run_once(i, cold_start=cold_start))

    aggregate = aggregate_runs(run_results)

    out_path = LOG_DIR / "baseline_comparison_repeated_summary.json"
    out_payload = {
        "runs": run_results,
        "aggregate": aggregate,
        "cold_start": cold_start,
    }
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== AGGREGATED JSON =====\n")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print_table(aggregate)
    print(f"\nSaved repeated-run summary to: {out_path}")


if __name__ == "__main__":
    main()