import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.create_bert_dataset import INTENT_LABELS

MODEL_DIR = "MedMind-BERT"

TEST_TEXTS = [
    "my chest hurts when I move",
    "What painkiller should I use for back pain?",
    "I want to book an appointment because my symptoms are not improving",
    "What can I do to ease my sore throat?",
    "What can I eat that will not upset my stomach?",
    "I am having trouble breathing and it is getting worse",
    "I am trying to make health a bigger priority",
]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def run_predictions():
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    tokens = tokenizer(
        TEST_TEXTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**tokens)

    predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    for text, prediction in zip(TEST_TEXTS, predictions):
        label = INTENT_LABELS.get(int(prediction), "general_health")
        print(f"{text} -> {label}")


if __name__ == "__main__":
    run_predictions()
