import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import config
from backend.rag_graph import build_rag_graph


app = FastAPI(title="Chatbot RAG - Turismo no Brasil")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    rag_graph = build_rag_graph()
    STARTUP_ERROR = None
except FileNotFoundError as exc:
    rag_graph = None
    STARTUP_ERROR = str(exc)


SESSIONS: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    if STARTUP_ERROR:
        return {"status": "error", "detail": STARTUP_ERROR}
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if rag_graph is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Índice vetorial não encontrado. Execute "
                "'python backend/ingest.py' antes de usar o chat. "
                f"Detalhe: {STARTUP_ERROR}"
            ),
        )

    session_id = req.session_id or str(uuid.uuid4())
    history = SESSIONS.setdefault(session_id, [])

    result = rag_graph.invoke(
        {
            "question": req.message,
            "chat_history": history,
        }
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    history.append({
        "role": "user",
        "content": req.message
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    SESSIONS[session_id] = history[-(config.MAX_HISTORY_TURNS * 2):]

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
    )


@app.get("/")
def serve_frontend():
    index_path = config.FRONTEND_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend não encontrado."
        )

    return FileResponse(index_path)


app.mount(
    "/static",
    StaticFiles(directory=str(config.FRONTEND_DIR)),
    name="static",
)