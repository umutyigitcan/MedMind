import json
from typing import Any, Dict, Optional

from app.appointment_service import create_appointment_from_message, request_appointment_datetime
from app.medical_answer_service import build_medical_answer
from app.retrieval_service import retrieve_medical_context


def get_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "answer_medical_question",
                "description": "Answer symptom, illness, medication, treatment, diet, and general health questions using medical retrieval context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's health-related question or symptom description.",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_appointment_datetime",
                "description": "Ask the user to provide appointment date and time when appointment details are missing.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_appointment",
                "description": "Create an appointment when the user provides a clear date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_message": {
                            "type": "string",
                            "description": "The appointment request containing date and time.",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Current chat session id.",
                        },
                    },
                    "required": ["appointment_message", "session_id"],
                },
            },
        },
    ]


def parse_tool_arguments(arguments: str) -> Dict[str, Any]:
    if not arguments:
        return {}

    return json.loads(arguments)


def run_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    session_id: str,
) -> Optional[Any]:
    if tool_name == "answer_medical_question":
        question = arguments.get("question", "")
        retrieval_result = retrieve_medical_context(question)
        return build_medical_answer(
            question=question,
            context=retrieval_result["context"],
        )

    if tool_name == "request_appointment_datetime":
        return request_appointment_datetime()

    if tool_name == "create_appointment":
        appointment_message = arguments.get("appointment_message", "")
        return create_appointment_from_message(
            message=appointment_message,
            session_id=session_id,
        )

    return None
