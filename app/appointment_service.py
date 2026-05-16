from typing import Optional, Tuple

from openai import OpenAI

from app.config import settings
from app.database import add_appointment
from app.session_store import append_message

client = OpenAI(api_key=settings.openai_api_key)


APPOINTMENT_FORMAT_HINT = (
    "Lütfen randevu almak için gün ve saat bilgisini birlikte yazın. "
    "Örnek: 20.12.2025 19:45"
)


def request_appointment_datetime() -> str:
    return APPOINTMENT_FORMAT_HINT


def extract_appointment_datetime(message: str) -> Optional[Tuple[str, str]]:
    settings.validate_openai_settings()

    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract appointment date and time from the user message. "
                    "Return only this exact format if both date and time exist: DD.MM.YYYY-HH.MM. "
                    "If either date or time is missing, return only: MISSING"
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    result = (response.choices[0].message.content or "").strip()

    if result.upper() == "MISSING":
        return None

    if "-" not in result:
        return None

    date_part, time_part = result.split("-", 1)

    date_part = date_part.strip()
    time_part = time_part.strip()

    if not date_part or not time_part:
        return None

    return date_part, time_part


def create_appointment_from_message(
    message: str,
    session_id: str,
) -> str:
    appointment_datetime = extract_appointment_datetime(message)

    if appointment_datetime is None:
        answer = request_appointment_datetime()
        append_message(session_id, "assistant", answer)
        return answer

    date_part, time_part = appointment_datetime

    add_appointment(date_part, time_part)

    answer = f"Randevunuz {date_part} tarihinde, saat {time_part} olarak oluşturuldu."

    append_message(session_id, "assistant", answer)

    return answer
