# modules/evaluator.py
from llm import call_structured_response
from prompts import EVALUATOR_PROMPT
from schemas import EVALUATOR_SCHEMA


def run_evaluator(
    config,
    task_descriptor,
    learner_state,
    learner_submission,
    tutor_output,
    assistant_output,
    retrieved_context,
    interaction_history,
    execution_log="",
):
    payload = {
        "task_descriptor": task_descriptor,
        "learner_state": learner_state,
        "learner_submission": learner_submission,
        "execution_log": execution_log,
        "tutor_output": tutor_output,
        "assistant_output": assistant_output,
        "retrieved_context": retrieved_context,
        "retrieved_mistakes": learner_state.get("retrieved_mistakes", []),
        "interaction_history": interaction_history,
        "ground_truth_artifacts": task_descriptor.get("ground_truth_artifacts", []),
        "learning_objectives": task_descriptor.get("learning_objectives", []),
        "learner_rubric_criteria": ["Accuracy", "Clarity", "Completeness", "Ethics"],
        "system_checks": [
            "guidance_quality",
            "reliability",
            "policy_compliance",
            "safety_flags",
        ],
    }

    return call_structured_response(
        model=config.modules.model,
        system_prompt=EVALUATOR_PROMPT,
        payload=payload,
        schema_name="iottri_evaluator_output",
        schema=EVALUATOR_SCHEMA,
        temperature=config.modules.evaluator_temperature,
    )