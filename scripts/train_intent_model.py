import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from scripts.create_bert_dataset import INTENT_LABELS, build_dataset

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "MedMind-BERT"
RESULTS_DIR = "results"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=True,
            max_length=128,
        )

    return dataset.map(tokenize, batched=True)


def compute_metrics(prediction_output):
    predictions = np.argmax(prediction_output.predictions, axis=1)

    accuracy = accuracy_score(prediction_output.label_ids, predictions)
    f1 = f1_score(prediction_output.label_ids, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


def train_model():
    dataset = build_dataset()
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = tokenize_dataset(split_dataset, tokenizer)

    device = get_device()
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(INTENT_LABELS),
    ).to(device)

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=8,
        weight_decay=0.01,
        logging_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Intent model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train_model()
