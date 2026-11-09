"""LangGraph agent: route -> retrieve -> grade -> (web fallback -> grade -> rewrite) -> answer."""

from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.rag.llm import get_llm as llm
from app.rag.state import AgentState, EvidenceGrade, RouteDecision
from app.rag.vectorstore import get_retriever

settings = get_settings()

SYSTEM_ROLE = "You are an enterprise HR policy and employee support assistant."

INSUFFICIENT_ANSWER = (
    "I couldn't find enough reliable evidence in the company HR knowledge base or external "
    "search to answer confidently. Please contact the HR team or share more details."
)


@lru_cache
def web_search() -> TavilySearch:
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilySearch(
        tavily_api_key=settings.tavily_api_key,
        max_results=settings.web_max_results,
        include_answer=True,
    )


def _trace(state: AgentState, message: str) -> list[str]:
    return [*state["trace"], message]


def _generate(prompt: str) -> str:
    return llm().invoke(prompt).content


def _grade(question: str, evidence: str) -> str:
    grader = llm().with_structured_output(EvidenceGrade)
    return grader.invoke(f"""{SYSTEM_ROLE}
Question: {question}
Evidence:
{evidence}

Return "good" only if the evidence is sufficient and directly relevant to answer specifically;
otherwise "weak". Respond as JSON: {{"grade": "good"}} or {{"grade": "weak"}}.""").grade


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)


# ---------------------------------------------------------------- nodes

def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision)
    route = router.invoke(f"""{SYSTEM_ROLE}
Use "kb" for anything about company HR policy: leave, holidays, benefits, payroll, remote work,
attendance, onboarding, performance, expenses, travel, conduct or employee support.
Use "direct" only for greetings, thanks or small talk that needs no company knowledge.
Question: {state['question']}
Respond as JSON: {{"route": "kb"}} or {{"route": "direct"}}.""").route
    return {"source_used": route, "trace": _trace(state, f"Router → {route.upper()}")}


def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {"kb_docs": docs, "trace": _trace(state, f"KB retrieval → {len(docs)} chunks")}


def grade_kb(state: AgentState):
    grade = _grade(state["question"], _format_docs(state["kb_docs"]))
    return {"kb_grade": grade, "trace": _trace(state, f"KB evidence → {grade.upper()}")}


def search_web(state: AgentState):
    result = web_search().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if result.get("answer"):
        lines.append(f"Search answer: {result['answer']}")
    for item in result.get("results", []):
        lines.append(f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\nContent: {item.get('content', '')}")
        citations.append({"title": item.get("title") or item.get("url", ""), "url": item.get("url", ""), "type": "web"})
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "trace": _trace(state, f"Web search → {len(citations)} results"),
    }


def grade_web(state: AgentState):
    grade = _grade(state["question"], state["web_results"])
    return {"web_grade": grade, "trace": _trace(state, f"Web evidence → {grade.upper()}")}


def rewrite_query(state: AgentState):
    rewritten = _generate(f"""Rewrite this HR / employee-support question to improve knowledge-base
retrieval and web search. Preserve intent, add useful HR keywords, do not answer it.
Return only the rewritten query.
Question: {state['question']}""").strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": _trace(state, f"Query rewrite → {rewritten}"),
    }


def generate_from_kb(state: AgentState):
    answer = _generate(f"""{SYSTEM_ROLE}
Answer ONLY from the company HR knowledge base below. Be concise, practical and respectful.
Present steps clearly when relevant. Never invent policy details.

Question: {state['question']}

Knowledge base:
{_format_docs(state['kb_docs'])}""")
    sources = dict.fromkeys(d.metadata.get("source", "Knowledge base") for d in state["kb_docs"])
    citations = [{"title": s.replace("\\", "/").split("/")[-1], "url": "", "type": "kb"} for s in sources]
    return {
        "answer": answer,
        "source_used": "knowledge_base",
        "citations": citations,
        "trace": _trace(state, "Answer → KNOWLEDGE BASE"),
    }


def generate_from_web(state: AgentState):
    answer = _generate(f"""{SYSTEM_ROLE}
The company knowledge base did not cover this question. Answer ONLY from the web evidence below,
and state clearly that this is external public information that should be validated with HR
before being treated as company policy.

Question: {state['question']}

Web evidence:
{state['web_results']}""")
    return {"answer": answer, "source_used": "web_search", "trace": _trace(state, "Answer → WEB SEARCH")}


def direct_answer(state: AgentState):
    answer = _generate(f"{SYSTEM_ROLE} Respond briefly and naturally to: {state['question']}")
    return {"answer": answer, "source_used": "direct", "trace": _trace(state, "Answer → DIRECT")}


def insufficient(state: AgentState):
    return {
        "answer": INSUFFICIENT_ANSWER,
        "source_used": "insufficient_evidence",
        "citations": [],
        "trace": _trace(state, "Stopped → insufficient evidence"),
    }


# ---------------------------------------------------------------- edges

def after_route(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"


def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"


def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    return "rewrite_query" if state["retry_count"] < settings.max_retries else "insufficient"


def build_graph():
    graph = StateGraph(AgentState)
    for node in (route_question, retrieve_kb, grade_kb, search_web, grade_web, rewrite_query,
                 generate_from_kb, generate_from_web, direct_answer, insufficient):
        graph.add_node(node.__name__, node)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", after_route)
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb)
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web)
    graph.add_edge("rewrite_query", "retrieve_kb")
    for terminal in ("generate_from_kb", "generate_from_web", "direct_answer", "insufficient"):
        graph.add_edge(terminal, END)
    return graph.compile()


agent_graph = build_graph()


def initial_state(question: str, **overrides) -> AgentState:
    state: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return {**state, **overrides}


def ask(question: str) -> AgentState:
    return agent_graph.invoke(initial_state(question))
