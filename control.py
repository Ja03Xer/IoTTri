# control.py
from __future__ import annotations

from statistics import mean
from typing import Dict, List


def get_sliding_window(turn_logs: List[dict], window_size: int) -> List[dict]:
    if not turn_logs:
        return []
    return turn_logs[-window_size:]


def accuracy_mean(window_logs: List[dict]) -> float:
    scores = []
    for log in window_logs:
        criteria = log["evaluator_output"]["learner_evaluation"]["criteria"]
        for c in criteria:
            if c["criterion"] == "Accuracy":
                scores.append(c["score"])
    return mean(scores) if scores else 5.0


def has_safety_flag(window_logs: List[dict]) -> bool:
    for log in window_logs:
        flags = log["evaluator_output"]["system_evaluation"]["safety_flags"]
        if any(flags.values()):
            return True
    return False


def count_recent_safety_hits(turn_logs: List[dict], window_size: int) -> int:
    recent = turn_logs[-window_size:]
    hits = 0
    for log in recent:
        flags = log["evaluator_output"]["system_evaluation"]["safety_flags"]
        if any(flags.values()):
            hits += 1
    return hits


def learner_stalled(window_logs: List[dict], stall_threshold_m: int) -> bool:
    recent = window_logs[-stall_threshold_m:]
    if len(recent) < stall_threshold_m:
        return False

    return all(
        (
            len(log["learner_state_before"].get("progress_markers", [])) == 0
            and len(log["learner_state_before"].get("evidence_markers", [])) == 0
        )
        for log in recent
    )


def consecutive_low_accuracy(windows: List[List[dict]], threshold: float, n: int) -> bool:
    if len(windows) < n:
        return False
    tail = windows[-n:]
    return all(accuracy_mean(w) < threshold for w in tail)


def _append_unique_constraint(config, text: str) -> None:
    if text not in config.extra_prompt_constraints:
        config.extra_prompt_constraints.append(text)


def build_adaptation_record(event_type: str, reason: str, before: dict, after: dict, triggers: dict) -> dict:
    return {
        "type": event_type,
        "reason": reason,
        "before": before,
        "after": after,
        "triggers": triggers,
    }


def apply_self_enhancement(
    *,
    config,
    current_window: List[dict],
    historical_windows: List[List[dict]],
    learner_state,
) -> dict | None:
    if not current_window:
        return None

    current_accuracy = accuracy_mean(current_window)
    safety_present = has_safety_flag(current_window)
    stalled = learner_stalled(current_window, config.adaptation.stall_threshold_m)

    if safety_present:
        before = {
            "config": config.snapshot(),
            "hint_level": learner_state.hint_level,
        }

        config.modules.assistant_temperature = max(
            config.adaptation.assistant_temp_min,
            config.modules.assistant_temperature - 0.1,
        )

        _append_unique_constraint(
            config,
            "Require the learner to provide observed evidence before more detailed recommendations."
        )
        _append_unique_constraint(
            config,
            "Reduce operational specificity and prioritise scope reminders."
        )
        _append_unique_constraint(
            config,
            "Prefer defensive validation, analytical reasoning, and authorised laboratory framing."
        )

        after = {
            "config": config.snapshot(),
            "hint_level": learner_state.hint_level,
        }

        return build_adaptation_record(
            event_type="safety_override",
            reason="Detected safety flags in current sliding window.",
            before=before,
            after=after,
            triggers={
                "window_accuracy_mean": current_accuracy,
                "safety_present": safety_present,
                "stalled": stalled,
            },
        )

    if consecutive_low_accuracy(
        historical_windows + [current_window],
        threshold=config.adaptation.accuracy_threshold,
        n=config.adaptation.reliability_persistence_n,
    ):
        before = {
            "config": config.snapshot(),
            "hint_level": learner_state.hint_level,
        }

        config.retrieval.top_k = min(config.retrieval.top_k + 1, config.retrieval.max_top_k)
        config.retrieval.attack_surface_strict_filter = True

        config.modules.tutor_temperature = max(
            config.adaptation.tutor_temp_min,
            config.modules.tutor_temperature - 0.1,
        )
        config.modules.assistant_temperature = max(
            config.adaptation.assistant_temp_min,
            config.modules.assistant_temperature - 0.1,
        )

        _append_unique_constraint(
            config,
            "Prefer assumption checking and evidence validation before suggestions."
        )
        _append_unique_constraint(
            config,
            "When uncertain, ask the learner to verify observed artefacts before drawing conclusions."
        )

        after = {
            "config": config.snapshot(),
            "hint_level": learner_state.hint_level,
        }

        return build_adaptation_record(
            event_type="reliability_improvement",
            reason="Persistently low Accuracy across consecutive sliding windows.",
            before=before,
            after=after,
            triggers={
                "window_accuracy_mean": current_accuracy,
                "safety_present": safety_present,
                "stalled": stalled,
            },
        )

    if stalled:
        before = {
            "config": config.snapshot(),
            "hint_level": learner_state.hint_level,
        }

        learner_state.hint_level = min(
            learner_state.hint_level + 1,
            config.adaptation.max_hint_level,
        )

        _append_unique_constraint(
            config,
            "Provide structured micro-goals and shorter, targeted check questions."
        )
        _append_unique_constraint(
            config,
            "Break assistance into smaller evidence-seeking steps."
        )

        after = {
            "config": config.snapshot(),
            "hint_level": learner_state.hint_level,
        }

        return build_adaptation_record(
            event_type="stall_recovery",
            reason="No new evidence or progress markers across recent turns.",
            before=before,
            after=after,
            triggers={
                "window_accuracy_mean": current_accuracy,
                "safety_present": safety_present,
                "stalled": stalled,
            },
        )

    return None