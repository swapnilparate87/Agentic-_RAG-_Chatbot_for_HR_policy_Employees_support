from typing import Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel


class RouteDecision(BaseModel):
    route: Literal["kb", "direct"]


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"]


class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: list[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    trace: list[str]
    citations: list[dict]
