from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime


AttackSurface = Literal[
    "device_web_interface",
    "local_network_services",
    "wireless_interfaces",
    "cloud_companion_api",
    "firmware_storage_and_update",
]

AssessmentPhase = Literal[
    "reconnaissance",
    "vulnerability_analysis",
    "validation",
    "reporting",
]


@dataclass
class LearningObjective:
    bloom_level: Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    objective: str


@dataclass
class GroundTruthArtifact:
    name: str
    evidence_type: str
    description: str


@dataclass
class TaskDescriptor:
    task_id: str
    scenario_identifier: str
    device_type: str
    scenario_summary: str
    exposed_protocols_interfaces: List[str]
    allowed_tool_categories: List[str]
    learning_objectives: List[LearningObjective]
    ground_truth_artifacts: List[GroundTruthArtifact]
    active_attack_surfaces: List[AttackSurface]
    laboratory_scope: str
    authorised_only: bool = True


# @dataclass
# class LearnerState:
#     turn_index: int
#     current_assessment_phase: AssessmentPhase
#     hint_level: int
#     recent_action_summary: str
#     confidence_indicators: Dict[str, Any] = field(default_factory=dict)
#     evidence_markers: List[str] = field(default_factory=list)
#     progress_markers: List[str] = field(default_factory=list)


@dataclass
class RetrievedSnippet:
    snippet_id: str
    attack_surface: AttackSurface
    title: str
    text: str
    source: str
    score: float


@dataclass
class TutorOutput:
    hint: str
    rationale: str
    check_question: str


@dataclass
class AssistantOutput:
    suggestion: str
    explanation: str
    safety_note: str


@dataclass
class EvaluatorCriterion:
    criterion: Literal["Accuracy", "Clarity", "Completeness", "Ethics"]
    score: int
    comment: str
    recommendation: str


@dataclass
class LearnerEvaluation:
    criteria: List[EvaluatorCriterion]
    feedback_summary: str
    improvement_guidance: str
    bloom_level: str


@dataclass
class EvaluatorSafetyFlags:
    unsafe_specificity: bool
    hallucination_risk: bool
    out_of_scope_request: bool
    malicious_intent_signal: bool


@dataclass
class SystemEvaluation:
    guidance_quality_summary: str
    reliability_comment: str
    policy_compliance_comment: str
    safety_flags: EvaluatorSafetyFlags


@dataclass
class EvaluatorOutput:
    learner_evaluation: LearnerEvaluation
    system_evaluation: SystemEvaluation


@dataclass
class TurnLog:
    timestamp_utc: str
    task_descriptor: Dict[str, Any]
    learner_state_before: Dict[str, Any]
    learner_state_after: Dict[str, Any]
    learner_query: str
    execution_log: str
    moderation_result: Dict[str, Any]
    retrieved_context: List[Dict[str, Any]]
    tutor_output: Dict[str, Any]
    assistant_output: Dict[str, Any]
    evaluator_output: Dict[str, Any]
    config_snapshot: Dict[str, Any]
    adaptation_event: Optional[Dict[str, Any]] = None

    @staticmethod
    def now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"


# models.py
from dataclasses import dataclass, field

@dataclass
class LearnerState:
    turn_index: int
    phase: str
    hint_level: int
    recent_action_summary: str
    confidence_indicators: list[str] = field(default_factory=list)
    evidence_markers: list[str] = field(default_factory=list)
    progress_markers: list[str] = field(default_factory=list)
    retrieved_mistakes: list[dict] = field(default_factory=list)