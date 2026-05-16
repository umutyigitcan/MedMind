from typing import Dict, List

from openai import OpenAI

from app.config import settings
from app.retrieval_service import retrieve_medical_context

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """
You are MedMind, a safe medical support assistant.

Your role:
- Explain possible general causes based on the provided medical context.
- Give safe, practical, non-diagnostic guidance.
- Encourage professional medical evaluation when symptoms are severe, persistent, unusual, or worsening.
- For emergency symptoms, clearly advise urgent medical help.
- Do not claim to diagnose the user.
- Do not prescribe medication doses.
- Do not replace a doctor.
- Keep the answer clear, calm, and useful.

Response style:
- Use simple language.
- Be concise but helpful.
- Mention warning signs when relevant.
- If the retrieved context is not enough, say that clearly.
"""


def build_medical_prompt(
    user_message: str,
    intent: str,
    context: str,
) -> str:
    return f"""
User message:
{user_message}

Detected intent:
{intent}

Retrieved medical context:
{context}

Generate a safe and helpful medical support answer based on the information above.
"""


def generate_medical_answer(
    user_message: str,
    intent: str,
    context: str,
) -> str:
    settings.validate_openai_settings()

    prompt = build_medical_prompt(
        user_message=user_message,
        intent=intent,
        context=context,
    )

    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content or ""


def answer_with_rag(
    user_message: str,
    intent: str,
    top_k: int = 3,
) -> Dict[str, object]:
    retrieved = retrieve_medical_context(
        query=user_message,
        top_k=top_k,
    )

    answer = generate_medical_answer(
        user_message=user_message,
        intent=intent,
        context=retrieved["context"],
    )

    return {
        "answer": answer,
        "intent": intent,
        "matches": retrieved["matches"],
    }


def build_emergency_answer() -> str:
    return (
        "Bu anlattığınız durum acil olabilir. Özellikle nefes darlığı, şiddetli göğüs ağrısı, "
        "bayılacak gibi olma, bilinç bulanıklığı, konuşma bozulması veya ağrının kola/çeneye yayılması "
        "gibi belirtiler varsa vakit kaybetmeden acil sağlık hizmeti alın. Ben burada kesin tanı koyamam; "
        "bu belirtiler profesyonel değerlendirme gerektirir."
    )
