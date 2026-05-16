from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from app.config import settings
from app.intent_service import predict_intent
from app.medical_answer_service import build_emergency_answer
from app.schemas import ChatRequest
from app.session_store import append_message, get_or_create_session
from app.tool_service import get_tool_definitions, parse_tool_arguments, run_tool_call
from app.vector_store import get_vector_store_stats


settings.validate_openai_settings()

client = OpenAI(api_key=settings.openai_api_key)

app = FastAPI(
    title="MedMind API",
    version="1.0.0",
    description="Medical assistant backend with intent classification, RAG retrieval and appointment scheduling.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "MedMind API",
        "chat_model": settings.openai_chat_model,
        "embedding_model": settings.openai_embedding_model,
    }


@app.get("/health")
def health_check():
    try:
        stats = get_vector_store_stats()

        return {
            "status": "healthy",
            "service": "MedMind API",
            "index_vectors": stats["index_vectors"],
            "metadata_count": stats["metadata_count"],
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store file missing: {str(error)}",
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(error)}",
        )


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id.strip()
    user_message = req.message.strip()

    if not session_id:
        raise HTTPException(status_code=400, detail="Session id cannot be empty.")

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        messages = get_or_create_session(session_id)
        append_message(session_id, "user", user_message)

        intent = predict_intent(user_message)

        if intent == "emergency_case":
            answer = build_emergency_answer()
            append_message(session_id, "assistant", answer)

            return {
                "answer": answer,
                "intent": intent,
                "source": "emergency_safety_rule",
            }

        messages.append(
            {
                "role": "system",
                "content": (
                    f"Detected user intent: {intent}. "
                    "Respond safely. Do not provide diagnosis. "
                    "For symptoms, medication, treatment, diet or general health questions, use the available tool. "
                    "For appointment requests, use the appointment tools."
                ),
            }
        )

        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            temperature=0.2,
            tools=get_tool_definitions(),
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_arguments = parse_tool_arguments(tool_call.function.arguments)

            tool_result = run_tool_call(
                tool_name=tool_name,
                arguments=tool_arguments,
                session_id=session_id,
            )

            if tool_result is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unsupported tool call: {tool_name}",
                )

            append_message(session_id, "assistant", str(tool_result))

            return {
                "answer": tool_result,
                "intent": intent,
                "source": tool_name,
            }

        answer = assistant_message.content or "I could not generate a response."
        append_message(session_id, "assistant", answer)

        return {
            "answer": answer,
            "intent": intent,
            "source": "chat_model",
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Required local resource is missing: {str(error)}",
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Chat request failed: {str(error)}",
        )
