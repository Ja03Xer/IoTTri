# modules/assistant.py
from llm import call_structured_response
from prompts import ASSISTANT_PROMPT
from schemas import ASSISTANT_SCHEMA


def run_assistant(
    config,
    task_descriptor,
    learner_state,
    learner_query,
    execution_log,
    retrieved_context,
):
    payload = {
        "task_descriptor": task_descriptor,
        "learner_state": learner_state,
        "learner_query": learner_query,
        "execution_log": execution_log,
        "retrieved_context": retrieved_context,
        "retrieved_mistakes": learner_state.get("retrieved_mistakes", []),
        "extra_prompt_constraints": config.extra_prompt_constraints,
    }

    return call_structured_response(
        model=config.modules.model,
        system_prompt=ASSISTANT_PROMPT,
        payload=payload,
        schema_name="iottri_assistant_output",
        schema=ASSISTANT_SCHEMA,
        temperature=config.modules.assistant_temperature,
    )