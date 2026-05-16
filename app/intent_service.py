from functools import lru_cache
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings

INTENT_LABELS: List[str] = [
    "symptom_check",
    "medication_query",
    "appointment_request",
    "treatment_advice",
    "diet_advice",
    "emergency_case",
    "general_health",
]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@lru_cache(maxsize=1)
def load_intent_model():
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(settings.bert_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.bert_model_path,
    ).to(device)

    model.eval()

    return tokenizer, model, device


def predict_intent(text: str) -> str:
    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("Text cannot be empty.")

    tokenizer, model, device = load_intent_model()

    tokens = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        logits = model(**tokens).logits

    prediction = torch.argmax(logits, dim=1).item()

    if prediction < 0 or prediction >= len(INTENT_LABELS):
        return "general_health"

    return INTENT_LABELS[prediction]
