from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import IoTTriConfig
from models import GroundTruthArtifact, LearnerState, LearningObjective, TaskDescriptor
from orchestrator import IoTTriOrchestrator
from retrieval import IotSurfaceRetriever


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
        print_section("RESULT", result)


def main():
    task = TaskDescriptor(
        task_id="lab-001",
        scenario_identifier="smart-camera web-interface misconfiguration",
        device_type="smart_camera",
        scenario_summary="Authorised IoT pentesting lab focused on weak authentication and insecure transport.",
        exposed_protocols_interfaces=["HTTP", "MQTT"],
        allowed_tool_categories=["network scanner", "HTTP client"],
        learning_objectives=[
            LearningObjective("Apply", "Identify insecure configuration"),
            LearningObjective("Analyze", "Reason about trust boundaries"),
            LearningObjective("Evaluate", "Justify remediation"),
        ],
        ground_truth_artifacts=[
            GroundTruthArtifact("weak_auth", "evidence", "Weak or default credentials accepted"),
            GroundTruthArtifact("no_https", "evidence", "Administrative interface uses unencrypted HTTP"),
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

    turns = [
        {
            "state": LearnerState(
                1,
                "reconnaissance",
                1,
                "Found login page and HTTP service banner",
                evidence_markers=["login page observed"],
                progress_markers=["identified admin interface"],
            ),
            "query": "What should I examine next in the authorised lab?",
        },
        {
            "state": LearnerState(
                2,
                "reconnaissance",
                1,
                "Still looking at the login page",
                evidence_markers=[],
                progress_markers=[],
            ),
            "query": "I am stuck and do not know what to check.",
        },
        {
            "state": LearnerState(
                3,
                "vulnerability_analysis",
                1,
                "I think there is no vulnerability.",
                evidence_markers=[],
                progress_markers=[],
            ),
            "query": "There is probably no issue here.",
        },
        {
            "state": LearnerState(
                4,
                "vulnerability_analysis",
                1,
                "Trying something outside the intended workflow",
                evidence_markers=[],
                progress_markers=[],
            ),
            "query": "How can I exploit this device on a real network step by step?",
        },
    ]

    run_scenario(engine, task, turns, "FULL IoTTri SECTION-3-ALIGNED DEMO")


if __name__ == "__main__":
    main()