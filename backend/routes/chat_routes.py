from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid
import json

try:
    from backend.database import (
        create_session,
        get_all_sessions,
        get_session_messages,
        add_message,
        update_session_title,
        delete_session
    )
    from backend.services.kaggle_client import KaggleClient
except ImportError:
    from database import (
        create_session,
        get_all_sessions,
        get_session_messages,
        add_message,
        update_session_title,
        delete_session
    )
    from services.kaggle_client import KaggleClient

router = APIRouter(prefix="/api/chat", tags=["chat"])

class CreateSessionPayload(BaseModel):
    title: Optional[str] = "New Conversation"
    model_repo: Optional[str] = ""
    quant_file: Optional[str] = ""

class SendMessagePayload(BaseModel):
    session_id: str
    content: str
    tunnel_url: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 1024
    reasoning: Optional[str] = "off"

@router.get("/sessions")
def list_sessions():
    return get_all_sessions()

@router.post("/sessions")
def new_session(payload: CreateSessionPayload):
    session_id = str(uuid.uuid4())
    title = payload.title or "New Chat Thread"
    session = create_session(
        session_id=session_id,
        title=title,
        model_repo=payload.model_repo or "",
        quant_file=payload.quant_file or ""
    )
    return session

class UpdateSessionTitlePayload(BaseModel):
    title: str

@router.patch("/sessions/{session_id}")
def update_title(session_id: str, payload: UpdateSessionTitlePayload):
    return update_session_title(session_id, payload.title)

@router.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}

@router.post("/stream")
async def stream_chat(payload: SendMessagePayload):
    if not payload.tunnel_url.strip():
        raise HTTPException(status_code=400, detail="Tunnel URL is required to communicate with Kaggle GPU.")

    add_message(payload.session_id, "user", payload.content)

    db_history = get_session_messages(payload.session_id)
    messages_payload = [{"role": msg["role"], "content": msg["content"]} for msg in db_history]

    client = KaggleClient(payload.tunnel_url)

    async def response_proxy():
        full_reply = ""
        async for chunk in client.stream_chat(
            messages=messages_payload,
            temperature=payload.temperature or 0.7,
            top_p=payload.top_p or 0.95,
            top_k=payload.top_k or 40,
            max_tokens=payload.max_tokens or 1024,
            reasoning=payload.reasoning or "off"
        ):
            if chunk.startswith("data: "):
                try:
                    data_str = chunk[6:].strip()
                    parsed = json.loads(data_str)
                    token = parsed.get("token", "")
                    if token:
                        full_reply += token
                except Exception:
                    pass
            yield chunk

        if full_reply.strip():
            add_message(payload.session_id, "assistant", full_reply.strip())

    return StreamingResponse(response_proxy(), media_type="text/event-stream")
