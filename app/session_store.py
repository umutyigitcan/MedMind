from typing import Dict, List


SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are MedMind, a careful and safe health assistant. "
        "You can provide general health information, but you do not diagnose. "
        "For emergency symptoms, advise the user to seek urgent medical help."
    ),
}

_SESSIONS: Dict[str, List[Dict[str, str]]] = {}


def get_or_create_session(session_id: str) -> List[Dict[str, str]]:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = [SYSTEM_MESSAGE.copy()]

    return _SESSIONS[session_id]


def append_message(session_id: str, role: str, content: str) -> List[Dict[str, str]]:
    messages = get_or_create_session(session_id)
    messages.append({"role": role, "content": content})

    return messages


def reset_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)
