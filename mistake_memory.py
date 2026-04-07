# mistake_memory.py
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from llm import get_embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class MistakeRecord:
    turn_index: int
    phase: str
    learner_query: str
    recent_action_summary: str
    attack_surfaces: list[str] = field(default_factory=list)
    retrieved_context_titles: list[str] = field(default_factory=list)
    feedback_summary: str = ""
    improvement_guidance: str = ""
    accuracy_score: int = 0
    timestamp_utc: str = ""
    embedding: list[float] = field(default_factory=list)

    def to_prompt_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "phase": self.phase,
            "learner_query": self.learner_query,
            "recent_action_summary": self.recent_action_summary,
            "attack_surfaces": self.attack_surfaces,
            "retrieved_context_titles": self.retrieved_context_titles,
            "feedback_summary": self.feedback_summary,
            "improvement_guidance": self.improvement_guidance,
            "accuracy_score": self.accuracy_score,
            "timestamp_utc": self.timestamp_utc,
        }


class MistakeMemory:
    def __init__(self, memory_path: str, embedding_model: str):
        self.memory_path = Path(memory_path)
        self.embedding_model = embedding_model
        self.records: List[MistakeRecord] = []
        self.load()

    def load(self):
        self.records = []

        if not self.memory_path.exists():
            return

        with self.memory_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    record = MistakeRecord(**obj)
                    self.records.append(record)
                except json.JSONDecodeError:
                    print(
                        f"[MistakeMemory] Skipping invalid JSONL line "
                        f"{line_no} in {self.memory_path}"
                    )
                    continue
                except TypeError as e:
                    print(
                        f"[MistakeMemory] Skipping malformed record at line "
                        f"{line_no} in {self.memory_path}: {e}"
                    )
                    continue

    def save_record(self, record: MistakeRecord) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self.records.append(record)

    def _build_record_text(
        self,
        *,
        phase: str,
        learner_query: str,
        recent_action_summary: str,
        attack_surfaces: list[str],
        retrieved_context_titles: list[str],
        feedback_summary: str,
        improvement_guidance: str,
    ) -> str:
        return "\n".join(
            [
                f"Phase: {phase}",
                f"Learner query: {learner_query}",
                f"Recent action summary: {recent_action_summary}",
                f"Attack surfaces: {', '.join(attack_surfaces)}",
                f"Retrieved context titles: {', '.join(retrieved_context_titles)}",
                f"Evaluator feedback summary: {feedback_summary}",
                f"Improvement guidance: {improvement_guidance}",
            ]
        )

    def maybe_store_mistake(
        self,
        *,
        turn_index: int,
        timestamp_utc: str,
        phase: str,
        learner_query: str,
        recent_action_summary: str,
        attack_surfaces: list[str],
        retrieved_context_titles: list[str],
        evaluator_output: dict,
    ) -> MistakeRecord | None:
        criteria = evaluator_output["learner_evaluation"]["criteria"]

        accuracy_score = None
        for c in criteria:
            if c["criterion"] == "Accuracy":
                accuracy_score = int(c["score"])
                break

        if accuracy_score is None:
            return None

        safety_flags = evaluator_output["system_evaluation"]["safety_flags"]
        low_accuracy = accuracy_score <= 3
        safety_hit = any(bool(v) for v in safety_flags.values())

        if not low_accuracy and not safety_hit:
            return None

        feedback_summary = evaluator_output["learner_evaluation"].get("feedback_summary", "").strip()
        improvement_guidance = evaluator_output["learner_evaluation"].get("improvement_guidance", "").strip()

        text = self._build_record_text(
            phase=phase,
            learner_query=learner_query,
            recent_action_summary=recent_action_summary,
            attack_surfaces=attack_surfaces,
            retrieved_context_titles=retrieved_context_titles,
            feedback_summary=feedback_summary,
            improvement_guidance=improvement_guidance,
        )
        emb = get_embedding(text, self.embedding_model)

        record = MistakeRecord(
            turn_index=turn_index,
            phase=phase,
            learner_query=learner_query,
            recent_action_summary=recent_action_summary,
            attack_surfaces=list(attack_surfaces),
            retrieved_context_titles=list(retrieved_context_titles),
            feedback_summary=feedback_summary,
            improvement_guidance=improvement_guidance,
            accuracy_score=accuracy_score,
            timestamp_utc=timestamp_utc,
            embedding=emb,
        )
        self.save_record(record)
        return record

    def retrieve_similar(
        self,
        *,
        phase: str,
        learner_query: str,
        recent_action_summary: str,
        attack_surfaces: list[str],
        top_k: int = 2,
        min_score: float = 0.35,
    ) -> list[dict]:
        if not self.records:
            return []

        query_text = "\n".join(
            [
                f"Phase: {phase}",
                f"Learner query: {learner_query}",
                f"Recent action summary: {recent_action_summary}",
                f"Attack surfaces: {', '.join(attack_surfaces)}",
            ]
        )
        query_emb = get_embedding(query_text, self.embedding_model)

        scored = []
        for record in self.records:
            score = cosine_similarity(query_emb, record.embedding)
            if score < min_score:
                continue
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, record in scored[:top_k]:
            obj = record.to_prompt_dict()
            obj["similarity_score"] = score
            results.append(obj)
        return results