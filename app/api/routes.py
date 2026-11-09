from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.workflow import ask
from app.services.audit import write_audit
from app.services.ingestion import SUPPORTED, ingest_files

router = APIRouter(prefix="/api")
settings = get_settings()


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


class Citation(BaseModel):
    title: str
    url: str
    type: str


class ChatResponse(BaseModel):
    answer: str
    source_used: str
    rewritten_query: str
    trace: list[str]
    citations: list[Citation]


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    write_audit(payload.question, result["source_used"], result["trace"])
    return ChatResponse(
        answer=result["answer"],
        source_used=result["source_used"],
        rewritten_query=result["current_query"],
        trace=result["trace"],
        citations=result["citations"],
    )


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported types: {', '.join(sorted(SUPPORTED))}")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / name
    dest.write_bytes(await file.read())
    return {"file": name, "chunks": ingest_files([dest])}
