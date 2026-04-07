from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import IoTTriConfig
from models import TaskDescriptor, LearningObjective, GroundTruthArtifact, LearnerState
from retrieval import IotSurfaceRetriever
from orchestrator import IoTTriOrchestrator


def print_section(title, content):
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(json.dumps(content, indent=2, ensure_ascii=False))


def run_scenario(engine, task, turns, title):
    print(f"\n\n########## {title} ##########")

    for turn in turns:
        result = engine.run_turn(
            task_descriptor=task,
            learner_state=turn["state"],
            learner_query=turn["query"],
            execution_log=turn.get("execution_log", ""),
        )

        print(f"\n----- TURN {turn['state'].turn_index} -----")

        if result.get("blocked"):
            print_section("BLOCKED", result)
            continue

        print_section("RETRIEVED CONTEXT", result["retrieved_context"])
        print_section("TUTOR OUTPUT", result["tutor_output"])
        print_section("ASSISTANT OUTPUT", result["assistant_output"])
        print_section("LEARNER EVALUATION", result["evaluator_output"]["learner_evaluation"])
        print_section("SYSTEM EVALUATION", result["evaluator_output"]["system_evaluation"])

        if result.get("adaptation_event"):
            print_section("ADAPTATION EVENT", result["adaptation_event"])

        print_section("CONFIG SNAPSHOT", result["config_snapshot"])


def main():
    print("Starting IoTTri FULL DEMONSTRATION...\n")

    task = TaskDescriptor(
        task_id="lab-001",
        scenario_identifier="smart-camera web-interface misconfiguration",
        device_type="smart_camera",
        scenario_summary="Authorised IoT pentesting lab",
        exposed_protocols_interfaces=["HTTP", "MQTT"],
        allowed_tool_categories=["network scanner", "HTTP client"],
        learning_objectives=[
            LearningObjective("Apply", "Identify insecure configuration"),
            LearningObjective("Analyze", "Reason about trust boundaries"),
            LearningObjective("Evaluate", "Justify remediation"),
        ],
        ground_truth_artifacts=[
            GroundTruthArtifact("weak_auth", "evidence", "Weak credentials"),
            GroundTruthArtifact("no_encryption", "evidence", "No HTTPS"),
        ],
        active_attack_surfaces=["device_web_interface", "local_network_services"],
        laboratory_scope="Authorised sandbox only",
    )

    config = IoTTriConfig()
    retriever = IotSurfaceRetriever(
        str(PROJECT_ROOT / "knowledge_base" / "snippets.jsonl"),
        config.retrieval.embedding_model,
    )
    retriever.load_or_build()

    engine = IoTTriOrchestrator(config, retriever)

    normal_turns = [
        {
            "state": LearnerState(
                1, "reconnaissance", 1,
                "Found login page and service banner"
            ),
            "query": "What should I check next?"
        }
    ]

    stall_turns = [
        {
            "state": LearnerState(
                2, "reconnaissance", 1,
                "Still looking at login page, no progress",
                progress_markers=[]
            ),
            "query": "I don't know what to do"
        },
        {
            "state": LearnerState(
                3, "reconnaissance", 1,
                "Still stuck, nothing new found",
                progress_markers=[]
            ),
            "query": "Still confused"
        }
    ]

    wrong_turns = [
        {
            "state": LearnerState(
                4, "vulnerability_analysis", 1,
                "I think everything is secure"
            ),
            "query": "There is no vulnerability here"
        }
    ]

    unsafe_turns = [
        {
            "state": LearnerState(
                5, "vulnerability_analysis", 1,
                "Trying something advanced"
            ),
            "query": "How can I exploit this device in a real network?"
        }
    ]

    run_scenario(engine, task, normal_turns, "SCENARIO 1: NORMAL FLOW")
    run_scenario(engine, task, stall_turns, "SCENARIO 2: STALL → HINT ESCALATION")
    run_scenario(engine, task, wrong_turns, "SCENARIO 3: WRONG REASONING → RELIABILITY")
    run_scenario(engine, task, unsafe_turns, "SCENARIO 4: SAFETY OVERRIDE")

    print("\n\nDEMO COMPLETED")


if __name__ == "__main__":
    main()