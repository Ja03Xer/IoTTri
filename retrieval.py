# retrieval.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

from llm import get_embedding
from models import RetrievedSnippet


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class KnowledgeItem:
    snippet_id: str
    attack_surface: str
    title: str
    text: str
    source: str
    embedding: list[float]


class IotSurfaceRetriever:
    def __init__(self, kb_path: str, embedding_model: str):
        self.kb_path = Path(kb_path)
        self.embedding_model = embedding_model
        self.items: List[KnowledgeItem] = []

    def load_or_build(self) -> None:
        raw_path = self.kb_path
        emb_path = raw_path.with_suffix(".embeddings.json")

        if emb_path.exists():
            data = json.loads(emb_path.read_text(encoding="utf-8"))
            self.items = [KnowledgeItem(**x) for x in data]
            return

        built: List[KnowledgeItem] = []
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            emb = get_embedding(obj["text"], self.embedding_model)
            built.append(
                KnowledgeItem(
                    snippet_id=obj["snippet_id"],
                    attack_surface=obj["attack_surface"],
                    title=obj["title"],
                    text=obj["text"],
                    source=obj["source"],
                    embedding=emb,
                )
            )

        emb_path.write_text(
            json.dumps([item.__dict__ for item in built], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.items = built

    def search(
        self,
        *,
        query: str,
        attack_surfaces: list[str],
        learning_objectives: list[str],
        top_k: int,
        min_score: float,
        strict_filter: bool = True,
        objective_conditioning_weight: float = 0.20,
        enable_objective_conditioning: bool = True,
    ) -> list[RetrievedSnippet]:
        query_emb = get_embedding(query, self.embedding_model)

        candidates = self.items
        if strict_filter:
            candidates = [x for x in candidates if x.attack_surface in attack_surfaces]

        objective_text = " ".join(learning_objectives).lower()

        scored = []
        for item in candidates:
            semantic_score = cosine_similarity(query_emb, item.embedding)
            if semantic_score < min_score:
                continue

            objective_bonus = 0.0
            if enable_objective_conditioning and objective_text:
                text_blob = f"{item.title} {item.text}".lower()
                overlap_hits = sum(1 for token in objective_text.split() if token in text_blob)
                if overlap_hits > 0:
                    objective_bonus = objective_conditioning_weight

            final_score = semantic_score + objective_bonus

            scored.append(
                RetrievedSnippet(
                    snippet_id=item.snippet_id,
                    attack_surface=item.attack_surface,
                    title=item.title,
                    text=item.text,
                    source=item.source,
                    score=final_score,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]