from typing import Dict, List

Message = Dict[str, str]

_SESSIONS: Dict[str, List[Message]] = {}

DEFAULT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful and safe health assistant. "
        "You do not diagnose users. You provide general guidance and encourage professional care when needed."
    ),
}


def get_or_create_session(session_id: str) -> List[Message]:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = [DEFAULT_SYSTEM_MESSAGE.copy()]

    return _SESSIONS[session_id]


def add_message(session_id: str, role: str, content: str) -> List[Message]:
    messages = get_or_create_session(session_id)
    messages.append({"role": role, "content": content})

    return messages


def reset_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)
