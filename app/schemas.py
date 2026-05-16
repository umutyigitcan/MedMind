from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """
    Incoming chat request from a user.
    session_id keeps short-term conversation memory separated per user/session.
    """

    session_id: str = Field(..., min_length=1, description="Unique session identifier.")
    message: str = Field(..., min_length=1, description="User message.")


class ChatResponse(BaseModel):
    """
    Standard chat response returned by MedMind.
    """

    answer: str
    intent: Optional[str] = None
    source: Optional[str] = None


class AppointmentRequest(BaseModel):
    """
    Appointment request extracted from user message.
    """

    appointmentrequest: str


class AppointmentDateRequest(BaseModel):
    """
    Appointment request with session information.
    """

    appointmentrequest: str
    session_id: str


class AppointmentResponse(BaseModel):
    """
    Appointment confirmation response.
    """

    message: str
    day: Optional[str] = None
    time: Optional[str] = None


class RagRequest(BaseModel):
    """
    Request model for medical RAG search.
    """

    query: str = Field(..., min_length=1)


class RagChunk(BaseModel):
    """
    Retrieved medical context chunk.
    """

    id: int
    text: str


class RagResponse(BaseModel):
    """
    RAG response with assistant answer and retrieved context.
    """

    answer: str
    context: list[RagChunk] = []


class HealthResponse(BaseModel):
    """
    Basic backend health response.
    """

    status: str
    service: str
