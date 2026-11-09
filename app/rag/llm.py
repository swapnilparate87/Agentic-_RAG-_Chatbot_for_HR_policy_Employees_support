from functools import lru_cache

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.core.config import get_settings


@lru_cache
def get_llm(model: str | None = None, reasoning: bool | None = None) -> ChatNVIDIA:
    """NVIDIA-hosted chat model (build.nvidia.com), shared by the agent and the evaluation judge.

    Nemotron models reason before answering, and that reasoning counts toward `max_tokens`.
    Turning reasoning off makes agent calls roughly 20x faster; the judge keeps it on.
    """
    settings = get_settings()
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    reasoning = settings.nvidia_reasoning if reasoning is None else reasoning
    return ChatNVIDIA(
        model=model or settings.nvidia_model,
        api_key=settings.nvidia_api_key,
        temperature=0,
        max_tokens=settings.nvidia_max_tokens,
        model_kwargs={"chat_template_kwargs": {"enable_thinking": reasoning}},
    )
