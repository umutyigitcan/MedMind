from typing import Dict, Any

from openai import OpenAI

from app.config import settings
from app.retrieval_service import retrieve_medical_context


client = OpenAI(api_key=settings.openai_api_key)


def build_medical_answer(query: str) -> str:
    settings.validate_openai_settings()

    retrieval_result: Dict[str, Any] = retrieve_medical_context(query)
    context_text = retrieval_result.get("context", "")

    messages = [
        {
            "role": "system",
            "content": (
                "You are MedMind, a careful and safe health assistant. "
                "Use the provided medical context when it is relevant. "
                "Do not provide a definitive diagnosis. "
                "Do not prescribe medication. "
                "Encourage professional medical evaluation when symptoms are serious, persistent, or unclear."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{query}\n\n"
                f"Retrieved medical context:\n{context_text}\n\n"
                "Write a helpful, safe, and clear answer."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def build_emergency_answer() -> str:
    return (
        "Bu anlattığınız durum acil olabilir. Özellikle nefes darlığı, şiddetli göğüs ağrısı, "
        "bayılacak gibi olma, bilinç bulanıklığı, konuşma bozulması veya ağrının kola/çeneye yayılması "
        "gibi belirtiler varsa vakit kaybetmeden acil sağlık hizmeti alın. Ben burada kesin tanı koyamam; "
        "bu belirtiler profesyonel değerlendirme gerektirir."
    )
