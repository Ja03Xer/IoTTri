# schemas.py
TUTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "hint": {"type": "string"},
        "rationale": {"type": "string"},
        "check_question": {"type": "string"},
    },
    "required": ["hint", "rationale", "check_question"],
    "additionalProperties": False,
}

ASSISTANT_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestion": {"type": "string"},
        "explanation": {"type": "string"},
        "safety_note": {"type": "string"},
    },
    "required": ["suggestion", "explanation", "safety_note"],
    "additionalProperties": False,
}

EVALUATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "learner_evaluation": {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {
                                "type": "string",
                                "enum": ["Accuracy", "Clarity", "Completeness", "Ethics"]
                            },
                            "score": {"type": "integer", "minimum": 1, "maximum": 5},
                            "comment": {"type": "string"},
                            "recommendation": {"type": "string"},
                        },
                        "required": ["criterion", "score", "comment", "recommendation"],
                        "additionalProperties": False,
                    },
                    "minItems": 4,
                    "maxItems": 4,
                },
                "feedback_summary": {"type": "string"},
                "improvement_guidance": {"type": "string"},
                "bloom_level": {"type": "string"},
            },
            "required": [
                "criteria",
                "feedback_summary",
                "improvement_guidance",
                "bloom_level",
            ],
            "additionalProperties": False,
        },
        "system_evaluation": {
            "type": "object",
            "properties": {
                "guidance_quality_summary": {"type": "string"},
                "reliability_comment": {"type": "string"},
                "policy_compliance_comment": {"type": "string"},
                "safety_flags": {
                    "type": "object",
                    "properties": {
                        "unsafe_specificity": {"type": "boolean"},
                        "hallucination_risk": {"type": "boolean"},
                        "out_of_scope_request": {"type": "boolean"},
                        "malicious_intent_signal": {"type": "boolean"},
                    },
                    "required": [
                        "unsafe_specificity",
                        "hallucination_risk",
                        "out_of_scope_request",
                        "malicious_intent_signal",
                    ],
                    "additionalProperties": False,
                },
            },
            "required": [
                "guidance_quality_summary",
                "reliability_comment",
                "policy_compliance_comment",
                "safety_flags",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["learner_evaluation", "system_evaluation"],
    "additionalProperties": False,
}