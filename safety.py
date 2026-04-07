# safety.py
from __future__ import annotations
from pathlib import Path

import os
from typing import Dict

from openai import OpenAI


_PROJECT_ROOT = Path(__file__).resolve().parent

def load_api_key() -> str:
    """Load the API key from environment first, then a local api_key.txt file.

    The local file is intended for private development only and should be
    excluded from version control.
    """
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    key_path = _PROJECT_ROOT / "api_key.txt"
    if key_path.exists():
        key = key_path.read_text(encoding="utf-8").strip()
        if key:
            return key

    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY or create a local api_key.txt file in the project root."
    )


client = OpenAI(api_key=load_api_key())


REAL_WORLD_MARKERS = [
    "real network",
    "production",
    "internet-facing",
    "public target",
    "stealth",
    "persistence",
    "evade detection",
    "bypass edr",
    "live target",
]

PAYLOAD_MARKERS = [
    "payload",
    "shellcode",
    "dropper",
    "reverse shell",
    "weaponize",
    "exploit chain",
]

STEP_BY_STEP_MARKERS = [
    "step by step",
    "exact commands",
    "full exploit",
    "walk me through exploitation",
]


def moderate_text(text: str) -> dict:
    res = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    return res.results[0].model_dump()


def detect_risk_markers(text: str) -> Dict[str, bool]:
    lowered = text.lower()
    return {
        "real_world_targeting": any(marker in lowered for marker in REAL_WORLD_MARKERS),
        "payload_request": any(marker in lowered for marker in PAYLOAD_MARKERS),
        "step_by_step_request": any(marker in lowered for marker in STEP_BY_STEP_MARKERS),
    }


def scope_gate(task_descriptor: dict, learner_query: str, safety_policy) -> dict:
    if safety_policy.require_authorised_scope and not task_descriptor.get("authorised_only", True):
        return {
            "allow": False,
            "reason": "Task is not marked as an authorised sandbox activity.",
            "risk_markers": {},
        }

    markers = detect_risk_markers(learner_query)

    if safety_policy.block_real_world_targeting and markers["real_world_targeting"]:
        return {
            "allow": False,
            "reason": "Query appears to target real-world or non-laboratory activity.",
            "risk_markers": markers,
        }

    if safety_policy.block_payload_generation and markers["payload_request"]:
        return {
            "allow": False,
            "reason": "Query requests payload generation or weaponization content outside laboratory support boundaries.",
            "risk_markers": markers,
        }

    if safety_policy.prohibit_step_by_step_exploitation and markers["step_by_step_request"]:
        return {
            "allow": False,
            "reason": "Query requests step-by-step exploitation guidance, which is outside supported educational scope.",
            "risk_markers": markers,
        }

    return {
        "allow": True,
        "reason": "",
        "risk_markers": markers,
    }


def needs_human_escalation(
    *,
    evaluator_flags: Dict[str, bool],
    moderation_result: Dict[str, object],
    repeat_safety_hits: int,
    escalation_config,
) -> bool:
    if not escalation_config.enabled:
        return False

    if escalation_config.escalate_on_out_of_scope and evaluator_flags.get("out_of_scope_request", False):
        return True

    if escalation_config.escalate_on_malicious_intent and evaluator_flags.get("malicious_intent_signal", False):
        return True

    if escalation_config.escalate_on_repeat_safety_flags and repeat_safety_hits >= 2:
        return True

    flagged = moderation_result.get("flagged", False)
    if flagged:
        return True

    return False