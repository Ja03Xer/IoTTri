# config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SafetyPolicy:
    laboratory_only: bool = True
    require_authorised_scope: bool = True
    prohibit_step_by_step_exploitation: bool = True
    require_evidence_verification_on_risk: bool = True
    allow_human_escalation: bool = True
    block_real_world_targeting: bool = True
    block_payload_generation: bool = True
    immutable: bool = True


@dataclass
class RetrievalConfig:
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 4
    max_top_k: int = 8
    min_score: float = 0.20
    attack_surface_strict_filter: bool = True
    objective_conditioning_weight: float = 0.20
    enable_objective_conditioning: bool = True


@dataclass
class ModuleConfig:
    model: str = "gpt-4.1-mini"
    tutor_temperature: float = 0.5
    assistant_temperature: float = 0.7
    evaluator_temperature: float = 0.0


@dataclass
class AdaptationConfig:
    window_size_w: int = 3
    stall_threshold_m: int = 2
    reliability_persistence_n: int = 2
    accuracy_threshold: float = 3.0
    min_hint_level: int = 1
    max_hint_level: int = 4
    tutor_temp_min: float = 0.2
    tutor_temp_max: float = 0.8
    assistant_temp_min: float = 0.2
    assistant_temp_max: float = 0.8
    require_new_evidence_turns: int = 2


@dataclass
class EscalationConfig:
    enabled: bool = True
    escalate_on_out_of_scope: bool = True
    escalate_on_malicious_intent: bool = True
    escalate_on_repeat_safety_flags: bool = True


@dataclass
class ReproducibilityConfig:
    experiment_id: str = "iottri_prototype_v2"
    prompt_version: str = "2026-03-26"
    schema_version: str = "2026-03-26"
    code_version: str = "prototype-aligned-section3"
    save_manifest: bool = True

@dataclass
class MistakeMemoryConfig:
    enabled: bool = True
    memory_path: str = "logs/mistake_memory.jsonl"
    top_k: int = 2
    min_similarity: float = 0.25

@dataclass
class IoTTriConfig:
    safety: SafetyPolicy = field(default_factory=SafetyPolicy)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    modules: ModuleConfig = field(default_factory=ModuleConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    extra_prompt_constraints: List[str] = field(default_factory=list)
    mistake_memory: MistakeMemoryConfig = field(default_factory=MistakeMemoryConfig)

    def snapshot(self) -> Dict:
        return {
            "safety": self.safety.__dict__,
            "retrieval": self.retrieval.__dict__,
            "modules": self.modules.__dict__,
            "adaptation": self.adaptation.__dict__,
            "escalation": self.escalation.__dict__,
            "reproducibility": self.reproducibility.__dict__,
            "extra_prompt_constraints": list(self.extra_prompt_constraints),
        }





