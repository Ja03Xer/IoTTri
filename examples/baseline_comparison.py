from __future__ import annotations

import copy
import csv
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import IoTTriConfig
from llm import call_structured_response
from orchestrator import IoTTriOrchestrator
from retrieval import IotSurfaceRetriever
from modules.evaluator import run_evaluator
from examples.test_cases_effectiveness import build_test_cases


BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "response": {"type": "string"},
    },
    "required": ["response"],
    "additionalProperties": False,
}

BASELINE_PROMPT = """
You are a single-LLM baseline for an authorised sandboxed IoT penetration-testing learning task.

Purpose:
Provide one direct response to the learner query.

Rules:
- Stay within the authorised educational lab scope.
- Be helpful, concise, and technically grounded.
- Do not provide real-world offensive guidance.
- Do not produce step-by-step exploitation procedures.
- You do NOT have separate Tutor / Assistant / Evaluator roles.
- You do NOT use mistake memory.
- You do NOT adapt over time.
- Return only the required JSON field.

Return exactly:
{
  "response": "..."
}
""".strip()


def run_single_llm_baseline(config, task_descriptor, learner_state, learner_query) -> str:
    payload = {
        "task_descriptor": asdict(task_descriptor),
        "learner_state": asdict(learner_state),
        "learner_query": learner_query,
        "retrieved_context": [],
        "ground_truth_artifacts": [asdict(x) for x in task_descriptor.ground_truth_artifacts],
        "learning_objectives": [asdict(x) for x in task_descriptor.learning_objectives],
    }

    result = call_structured_response(
        model=config.modules.model,
        system_prompt=BASELINE_PROMPT,
        payload=payload,
        schema_name="single_llm_baseline_output",
        schema=BASELINE_SCHEMA,
        temperature=config.modules.assistant_temperature,
    )
    return result["response"]


def evaluate_baseline_with_iottri_evaluator(
    config,
    task_descriptor,
    learner_state,
    learner_query,
    baseline_response,
):
    dummy_tutor_output = {
        "hint": "",
        "rationale": "",
        "check_question": "",
    }
    dummy_assistant_output = {
        "suggestion": baseline_response,
        "explanation": "Direct response produced by the single-LLM baseline.",
        "safety_note": "Baseline evaluation in authorised sandbox context.",
    }

    return run_evaluator(
        config=config,
        task_descriptor=asdict(task_descriptor),
        learner_state=asdict(learner_state),
        learner_submission=learner_query,
        tutor_output=dummy_tutor_output,
        assistant_output=dummy_assistant_output,
        retrieved_context=[],
        interaction_history=[],
        execution_log="",
    )


def get_criterion_score(evaluator_output, criterion_name):
    for item in evaluator_output["learner_evaluation"]["criteria"]:
        if item["criterion"] == criterion_name:
            return item["score"]
    return None


def avg(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else 0.0


def run_one_case(case, iottri_engine, baseline_config):
    task = case["task"]
    state = copy.deepcopy(case["state"])
    query = case["query"]

    # IoTTri: shared orchestrator, so adaptation and memory can accumulate
    iottri_result = iottri_engine.run_turn(
        task_descriptor=task,
        learner_state=state,
        learner_query=query,
        execution_log="",
    )

    iottri_eval = None if iottri_result.get("blocked") else iottri_result["evaluator_output"]

    # Baseline: always stateless
    baseline_response = run_single_llm_baseline(
        config=baseline_config,
        task_descriptor=task,
        learner_state=copy.deepcopy(case["state"]),
        learner_query=query,
    )

    baseline_eval = evaluate_baseline_with_iottri_evaluator(
        config=baseline_config,
        task_descriptor=task,
        learner_state=copy.deepcopy(case["state"]),
        learner_query=query,
        baseline_response=baseline_response,
    )

    row = {
        "case_id": case["case_id"],
        "group": case.get("group", "unknown"),
        "query": query,
        "expected_focus": case.get("expected_focus", ""),
        "iottri_accuracy": None if iottri_eval is None else get_criterion_score(iottri_eval, "Accuracy"),
        "iottri_clarity": None if iottri_eval is None else get_criterion_score(iottri_eval, "Clarity"),
        "iottri_completeness": None if iottri_eval is None else get_criterion_score(iottri_eval, "Completeness"),
        "iottri_ethics": None if iottri_eval is None else get_criterion_score(iottri_eval, "Ethics"),
        "baseline_accuracy": get_criterion_score(baseline_eval, "Accuracy"),
        "baseline_clarity": get_criterion_score(baseline_eval, "Clarity"),
        "baseline_completeness": get_criterion_score(baseline_eval, "Completeness"),
        "baseline_ethics": get_criterion_score(baseline_eval, "Ethics"),
        "baseline_response": baseline_response,
    }
    return row


def build_summary(results):
    return {
        "IoTTri": {
            "Accuracy": avg([r["iottri_accuracy"] for r in results]),
            "Clarity": avg([r["iottri_clarity"] for r in results]),
            "Completeness": avg([r["iottri_completeness"] for r in results]),
            "Ethics": avg([r["iottri_ethics"] for r in results]),
        },
        "SingleLLMBaseline": {
            "Accuracy": avg([r["baseline_accuracy"] for r in results]),
            "Clarity": avg([r["baseline_clarity"] for r in results]),
            "Completeness": avg([r["baseline_completeness"] for r in results]),
            "Ethics": avg([r["baseline_ethics"] for r in results]),
        },
    }


def save_json(results, summary):
    output_dir = PROJECT_ROOT / "examples" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "effectiveness_vs_baseline_results.json"
    path.write_text(
        json.dumps({"summary": summary, "cases": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def save_csv(results):
    output_dir = PROJECT_ROOT / "examples" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "effectiveness_vs_baseline_results.csv"

    fieldnames = [
        "case_id",
        "group",
        "query",
        "expected_focus",
        "iottri_accuracy",
        "iottri_clarity",
        "iottri_completeness",
        "iottri_ethics",
        "baseline_accuracy",
        "baseline_clarity",
        "baseline_completeness",
        "baseline_ethics",
        "baseline_response",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})

    return path


def print_paper_table(summary):
    print("\n===== PAPER-READY SUMMARY TABLE =====\n")
    print("| System | Accuracy ↑ | Clarity ↑ | Completeness ↑ | Ethics ↑ |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| IoTTri | "
        f"{summary['IoTTri']['Accuracy']:.2f} | "
        f"{summary['IoTTri']['Clarity']:.2f} | "
        f"{summary['IoTTri']['Completeness']:.2f} | "
        f"{summary['IoTTri']['Ethics']:.2f} |"
    )
    print(
        f"| Single-LLM Baseline | "
        f"{summary['SingleLLMBaseline']['Accuracy']:.2f} | "
        f"{summary['SingleLLMBaseline']['Clarity']:.2f} | "
        f"{summary['SingleLLMBaseline']['Completeness']:.2f} | "
        f"{summary['SingleLLMBaseline']['Ethics']:.2f} |"
    )


def main():
    # Shared retriever
    bootstrap_config = IoTTriConfig()
    retriever = IotSurfaceRetriever(
        str(PROJECT_ROOT / "knowledge_base" / "snippets.jsonl"),
        bootstrap_config.retrieval.embedding_model,
    )
    retriever.load_or_build()

    # Shared IoTTri config and engine
    iottri_config = IoTTriConfig()
    iottri_config.mistake_memory.enabled = True
    iottri_config.mistake_memory.memory_path = str(
        PROJECT_ROOT / "examples" / "logs" / "mistake_memory.jsonl"
    )

    iottri_engine = IoTTriOrchestrator(
        config=iottri_config,
        retriever=retriever,
        log_dir=str(PROJECT_ROOT / "examples" / "logs" / "effectiveness_vs_baseline"),
    )

    # Baseline config: stateless
    baseline_config = IoTTriConfig()
    baseline_config.mistake_memory.enabled = False

    cases = build_test_cases()
    print(f"Running {len(cases)} test cases...\n")

    results = []
    for case in cases:
        print(f"Running case: {case['case_id']} ({case['group']})")
        results.append(run_one_case(case, iottri_engine, baseline_config))

    summary = build_summary(results)
    json_path = save_json(results, summary)
    csv_path = save_csv(results)

    print("\n===== SUMMARY (JSON) =====\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print_paper_table(summary)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()