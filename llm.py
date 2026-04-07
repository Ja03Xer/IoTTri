# llm.py
from __future__ import annotations
from pathlib import Path
import json
import os
from typing import Any, Dict, Optional

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


def call_structured_response(
    *,
    model: str,
    system_prompt: str,
    payload: Dict[str, Any],
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float,
) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
            },
        ],
        temperature=temperature,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        store=False,
    )
    return json.loads(response.output_text)


def get_embedding(text: str, model: str) -> list[float]:
    res = client.embeddings.create(model=model, input=text)
    return res.data[0].embedding


def get_runtime_metadata(model: str) -> Dict[str, Optional[str]]:
    return {
        "llm_model": model,
        "embedding_model_env": os.getenv("EMBEDDING_MODEL"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
    }