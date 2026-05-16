# MedMind

MedMind is an AI-powered health assistant backend that combines intent classification, medical document retrieval, safe answer generation, appointment handling, session memory, and PostgreSQL integration.

The project is designed as a modular backend system. It uses a BERT-based intent classifier to understand the user's health-related request, a FAISS-powered RAG pipeline to retrieve relevant medical document chunks, and OpenAI function calling to route the conversation to the correct backend service.

MedMind does not provide medical diagnosis. It is built as a safe health assistant prototype that gives general guidance, detects emergency-like cases, and encourages professional medical support when needed.

## Features

- BERT-based health intent classification
- Medical PDF loading and chunking
- OpenAI embedding generation
- FAISS vector index creation
- Metadata persistence for retrieved medical chunks
- RAG-based medical context retrieval
- OpenAI-powered safe response generation
- Emergency-aware response handling
- Appointment scheduling flow
- PostgreSQL appointment storage
- In-memory session history
- FastAPI backend API
- CORS support
- Modular service-based architecture
- Training and testing scripts for the BERT intent model

## Supported Intent Categories

MedMind classifies user messages into the following categories:

- symptom_check
- medication_query
- appointment_request
- treatment_advice
- diet_advice
- emergency_case
- general_health

## Tech Stack

- Python
- FastAPI
- OpenAI API
- FAISS
- PyMuPDF
- NumPy
- PostgreSQL
- psycopg2
- PyTorch
- Transformers
- BERT
- Pydantic
- Uvicorn
- RAG
- Function Calling

## Project Structure

```text
MedMind/
  app/
    __init__.py
    main.py
    config.py
    schemas.py
    database.py
    session_store.py
    intent_service.py
    pdf_service.py
    embedding_service.py
    index_builder.py
    metadata_store.py
    vector_store.py
    retrieval_service.py
    medical_answer_service.py
    appointment_service.py
    tool_service.py
  scripts/
    create_bert_dataset.py
    train_intent_model.py
    test_intent_model.py
  data/
    .gitkeep
  .gitignore
  requirements.txt
  README.md