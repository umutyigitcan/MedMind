from datasets import Dataset

INTENT_LABELS = {
    0: "symptom_check",
    1: "medication_query",
    2: "appointment_request",
    3: "treatment_advice",
    4: "diet_advice",
    5: "emergency_case",
    6: "general_health",
}

INTENT_EXAMPLES = {
    0: [
        "I have a really bad headache",
        "My chest hurts when I move",
        "I feel dizzy and nauseous",
        "My stomach hurts a lot",
        "I can't stop coughing",
        "My throat hurts when I swallow",
        "My lower back has been hurting all day",
        "My nose feels completely blocked",
        "I feel extremely tired today",
    ],
    1: [
        "What medicine should I take for this headache?",
        "Can I take ibuprofen for this pain?",
        "What can I take for my sore throat?",
        "Which medicine is good for nausea?",
        "Do I need antibiotics for this?",
        "What painkiller should I use for back pain?",
        "Can I use a nasal spray for this congestion?",
        "Is paracetamol okay for what I am feeling?",
    ],
    2: [
        "Can I schedule an appointment for tomorrow?",
        "I would like to book a doctor appointment",
        "I need to see a doctor today",
        "Can you arrange a visit for me?",
        "I want to schedule an appointment because my symptoms are not improving",
        "Can I come in tomorrow for my shoulder pain?",
        "I need a follow-up appointment for my chest discomfort",
    ],
    3: [
        "What should I do to reduce this pain?",
        "Is there anything I can do at home to feel better?",
        "How can I relieve this chest discomfort?",
        "What can I do to ease my sore throat?",
        "How can I manage this stomach pain at home?",
        "What helps with neck stiffness?",
        "How can I reduce this coughing at home?",
    ],
    4: [
        "What should I eat to feel lighter today?",
        "What foods are good for calming the stomach?",
        "Should I avoid spicy foods right now?",
        "What can I eat that will not upset my stomach?",
        "Is soup a good option when I feel sick?",
        "What foods help reduce nausea?",
        "Should I avoid caffeine for a bit?",
    ],
    5: [
        "I cannot breathe properly, what should I do right now?",
        "My chest hurts really badly, I think something is wrong",
        "I feel like I am going to pass out",
        "My chest pain is spreading to my left arm",
        "My throat feels like it is closing",
        "I can barely breathe",
        "My vision is getting blurry suddenly",
        "I feel like I am losing consciousness",
    ],
    6: [
        "I want to improve my daily routine for better health",
        "How can I keep my energy levels steady throughout the day?",
        "I have not been sleeping well",
        "I want to build healthier habits",
        "I am trying to drink more water every day",
        "Is walking every day enough for staying fit?",
        "I want to live a healthier lifestyle overall",
    ],
}


def build_records():
    texts = []
    labels = []

    for label, examples in INTENT_EXAMPLES.items():
        for example in examples:
            texts.append(example)
            labels.append(label)

    return {
        "text": texts,
        "label": labels,
    }


def build_dataset() -> Dataset:
    return Dataset.from_dict(build_records())


if __name__ == "__main__":
    dataset = build_dataset()
    print(dataset)
    print(INTENT_LABELS)
