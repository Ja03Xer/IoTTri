# modules/tutor.py
from llm import call_structured_response
from prompts import TUTOR_PROMPT
from schemas import TUTOR_SCHEMA


def run_tutor(config, task_descriptor, learner_state, retrieved_context):
    payload = {
        "task_descriptor": task_descriptor,
        "learner_state": learner_state,
        "retrieved_context": retrieved_context,
        "retrieved_mistakes": learner_state.get("retrieved_mistakes", []),
        "extra_prompt_constraints": config.extra_prompt_constraints,
    }

    return call_structured_response(
        model=config.modules.model,
        system_prompt=TUTOR_PROMPT,
        payload=payload,
        schema_name="iottri_tutor_output",
        schema=TUTOR_SCHEMA,
        temperature=config.modules.tutor_temperature,
    )