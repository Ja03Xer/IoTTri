# orchestrator.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from control import (
    apply_self_enhancement,
    count_recent_safety_hits,
    get_sliding_window,
)
from llm import get_runtime_metadata
from logging_utils import append_jsonl, build_manifest, write_json
from mistake_memory import MistakeMemory
from models import LearnerState, TurnLog
from modules.assistant import run_assistant
from modules.evaluator import run_evaluator
from modules.tutor import run_tutor
from retrieval import IotSurfaceRetriever
from safety import moderate_text, needs_human_escalation, scope_gate


class IoTTriOrchestrator:
    def __init__(self, config, retriever: IotSurfaceRetriever, log_dir: str = "logs"):
        self.config = config
        self.retriever = retriever
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.turn_logs: list[dict] = []
        self.windows: list[list[dict]] = []

        if self.config.mistake_memory.enabled:
            self.mistake_memory = MistakeMemory(
                memory_path=self.config.mistake_memory.memory_path,
                embedding_model=self.config.retrieval.embedding_model,
            )
        else:
            self.mistake_memory = None

        if self.config.reproducibility.save_manifest:
            manifest = build_manifest(
                experiment_id=self.config.reproducibility.experiment_id,
                code_version=self.config.reproducibility.code_version,
                prompt_version=self.config.reproducibility.prompt_version,
                schema_version=self.config.reproducibility.schema_version,
                config_snapshot=self.config.snapshot(),
                runtime_metadata=get_runtime_metadata(self.config.modules.model),
            )
            write_json(self.log_dir / "manifest.json", manifest)

    def _save_turn_log(self, turn_log: dict) -> None:
        turn_id = turn_log["learner_state_before"]["turn_index"]
        write_json(self.log_dir / f"turn_{turn_id:03d}.json", turn_log)
        append_jsonl(self.log_dir / "turns.jsonl", turn_log)

    def _build_interaction_history(self) -> list[dict]:
        history = []
        for log in self.turn_logs:
            history.append(
                {
                    "turn_index": log["learner_state_before"]["turn_index"],
                    "learner_query": log["learner_query"],
                    "execution_log": log["execution_log"],
                    "retrieved_context": log["retrieved_context"],
                    "tutor_output": log["tutor_output"],
                    "assistant_output": log["assistant_output"],
                    "evaluator_output": log["evaluator_output"],
                }
            )
        return history

    def _escalate_to_human_queue(self, turn_log: dict, reason: str) -> dict:
        record = {
            "turn_index": turn_log["learner_state_before"]["turn_index"],
            "timestamp_utc": turn_log["timestamp_utc"],
            "reason": reason,
            "learner_query": turn_log["learner_query"],
            "system_evaluation": turn_log["evaluator_output"]["system_evaluation"],
        }
        append_jsonl(self.log_dir / "human_escalation_queue.jsonl", record)
        return record

    def _safe_block_response(self, scope_result: dict, moderation_result: dict) -> dict:
        return {
            "blocked": True,
            "reason": scope_result["reason"],
            "moderation": moderation_result,
            "assistant_output": {
                "suggestion": "Refocus on the authorised laboratory task and provide evidence-based observations from the sandbox scenario.",
                "explanation": "The request falls outside the supported educational scope, so the system cannot provide operational guidance for it.",
                "safety_note": "Only authorised sandboxed laboratory activities are supported.",
            },
        }

    def run_turn(self, task_descriptor, learner_state: LearnerState, learner_query: str, execution_log: str = ""):
        task_dict = asdict(task_descriptor)
        learner_state_dict = asdict(learner_state)

        scope_result = scope_gate(task_dict, learner_query, self.config.safety)
        moderation_result = moderate_text(learner_query)

        if not scope_result["allow"]:
            return self._safe_block_response(scope_result, moderation_result)

        if self.config.mistake_memory.enabled:
            retrieved_mistakes = self.mistake_memory.retrieve_similar(
                phase=learner_state.phase,
                learner_query=learner_query,
                recent_action_summary=learner_state.recent_action_summary,
                attack_surfaces=task_descriptor.active_attack_surfaces,
                top_k=self.config.mistake_memory.top_k,
                min_score=self.config.mistake_memory.min_similarity,
            )
        else:
            retrieved_mistakes = []

        learner_state.retrieved_mistakes = retrieved_mistakes
        learner_state_dict = asdict(learner_state)

        retrieval_query = "\n".join(
            [
                task_descriptor.scenario_identifier,
                learner_query,
                learner_state.recent_action_summary,
                " ".join(lo.objective for lo in task_descriptor.learning_objectives),
            ]
        )

        retrieved = self.retriever.search(
            query=retrieval_query,
            attack_surfaces=task_descriptor.active_attack_surfaces,
            learning_objectives=[lo.objective for lo in task_descriptor.learning_objectives],
            top_k=self.config.retrieval.top_k,
            min_score=self.config.retrieval.min_score,
            strict_filter=self.config.retrieval.attack_surface_strict_filter,
            objective_conditioning_weight=self.config.retrieval.objective_conditioning_weight,
            enable_objective_conditioning=self.config.retrieval.enable_objective_conditioning,
        )
        retrieved_dicts = [asdict(x) for x in retrieved]

        tutor_output = run_tutor(
            self.config,
            task_dict,
            learner_state_dict,
            retrieved_dicts,
        )

        assistant_output = run_assistant(
            self.config,
            task_dict,
            learner_state_dict,
            learner_query,
            execution_log,
            retrieved_dicts,
        )

        interaction_history = self._build_interaction_history()

        evaluator_output = run_evaluator(
            self.config,
            task_dict,
            learner_state_dict,
            learner_query,
            tutor_output,
            assistant_output,
            retrieved_dicts,
            interaction_history=interaction_history,
            execution_log=execution_log,
        )

        safety_flags = evaluator_output["system_evaluation"]["safety_flags"]

        if any(safety_flags.values()):
            assistant_output = {
                "suggestion": "Refocus on analysing the authorised lab scenario and continue only with evidence-based, low-risk validation steps.",
                "explanation": "The evaluator detected safety or scope concerns, so guidance is being de-risked and kept within the educational sandbox.",
                "safety_note": "Only authorised sandboxed laboratory activities are supported, and further detail requires learner-provided evidence.",
            }

            self.config.retrieval.attack_surface_strict_filter = True
            self.config.retrieval.top_k = min(
                self.config.retrieval.top_k + 1,
                self.config.retrieval.max_top_k,
            )

            if "Require the learner to provide observed evidence before more detailed recommendations." not in self.config.extra_prompt_constraints:
                self.config.extra_prompt_constraints.append(
                    "Require the learner to provide observed evidence before more detailed recommendations."
                )

            if "Reduce operational specificity and prioritise scope reminders." not in self.config.extra_prompt_constraints:
                self.config.extra_prompt_constraints.append(
                    "Reduce operational specificity and prioritise scope reminders."
                )

        turn_timestamp = TurnLog.now_iso()

        if self.config.mistake_memory.enabled:
            self.mistake_memory.maybe_store_mistake(
                turn_index=learner_state.turn_index,
                timestamp_utc=turn_timestamp,
                phase=learner_state.phase,
                learner_query=learner_query,
                recent_action_summary=learner_state.recent_action_summary,
                attack_surfaces=task_descriptor.active_attack_surfaces,
                retrieved_context_titles=[x["title"] for x in retrieved_dicts],
                evaluator_output=evaluator_output,
            )

        turn_log = {
            "timestamp_utc": turn_timestamp,
            "task_descriptor": task_dict,
            "learner_state_before": learner_state_dict,
            "learner_state_after": asdict(learner_state),
            "learner_query": learner_query,
            "execution_log": execution_log,
            "moderation_result": moderation_result,
            "scope_result": scope_result,
            "retrieved_context": retrieved_dicts,
            "retrieved_mistakes": retrieved_mistakes,
            "interaction_history_used_by_evaluator": interaction_history,
            "tutor_output": tutor_output,
            "assistant_output": assistant_output,
            "evaluator_output": evaluator_output,
            "config_snapshot": self.config.snapshot(),
            "adaptation_event": None,
            "human_escalation": None,
        }

        self.turn_logs.append(turn_log)

        current_window = get_sliding_window(self.turn_logs, self.config.adaptation.window_size_w)

        adaptation_event = apply_self_enhancement(
            config=self.config,
            current_window=current_window,
            historical_windows=self.windows,
            learner_state=learner_state,
        )

        turn_log["adaptation_event"] = adaptation_event
        turn_log["learner_state_after"] = asdict(learner_state)
        turn_log["config_snapshot"] = self.config.snapshot()

        if len(current_window) == self.config.adaptation.window_size_w:
            self.windows.append(list(current_window))

        repeat_safety_hits = count_recent_safety_hits(
            self.turn_logs,
            self.config.adaptation.window_size_w,
        )

        if needs_human_escalation(
            evaluator_flags=safety_flags,
            moderation_result=moderation_result,
            repeat_safety_hits=repeat_safety_hits,
            escalation_config=self.config.escalation,
        ):
            turn_log["human_escalation"] = self._escalate_to_human_queue(
                turn_log,
                reason="High-risk or repeated safety issue detected.",
            )

        self._save_turn_log(turn_log)
        return turn_log