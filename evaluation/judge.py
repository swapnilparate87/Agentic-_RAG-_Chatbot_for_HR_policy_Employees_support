"""DeepEval judge backed by an NVIDIA-hosted model."""

import asyncio
import json
import re
import time

from deepeval.models import DeepEvalBaseLLM
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import BaseModel

from app.core.config import get_settings
from app.rag.llm import get_llm

ATTEMPTS = 5


def _backoff(attempt: int) -> float:
    return min(60, 5 * 2 ** attempt)  # 5s, 10s, 20s, 40s: long enough to recover from HTTP 429


def _parse(text: str, schema: type[BaseModel]) -> BaseModel:
    """Fallback for when guided JSON fails: pull the first JSON object out of free text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    match = re.search(r"\{.*\}", text, re.S)
    return schema.model_validate(json.loads(match.group(0) if match else text))


class NvidiaJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None, max_parallel: int = 4, requests_per_minute: int = 30):
        self.model_name = model or get_settings().nvidia_eval_model
        # NVIDIA's API allows ~40 requests/min per account; each DeepEval metric makes several calls.
        self._limiter = InMemoryRateLimiter(requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.2, max_bucket_size=2)
        self._semaphore = asyncio.Semaphore(max_parallel)  # stay under NVIDIA API rate limits
        self.usage = UsageMetadataCallbackHandler()  # token usage across all judge calls
        self._config = {"callbacks": [self.usage]}
        super().__init__(self.model_name)

    def load_model(self):
        return get_llm(self.model_name, reasoning=True).model_copy(update={"rate_limiter": self._limiter})

    def get_model_name(self) -> str:
        return self.model_name

    def supports_structured_outputs(self) -> bool:
        return True

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        for attempt in range(ATTEMPTS):
            try:
                if schema is None:
                    return self.model.invoke(prompt, config=self._config).content
                try:
                    return self.model.with_structured_output(schema).invoke(prompt, config=self._config)
                except Exception:
                    return _parse(self.model.invoke(prompt, config=self._config).content, schema)
            except Exception:
                if attempt == ATTEMPTS - 1:
                    raise
                time.sleep(_backoff(attempt))

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        async with self._semaphore:
            for attempt in range(ATTEMPTS):
                try:
                    if schema is None:
                        return (await self.model.ainvoke(prompt, config=self._config)).content
                    try:
                        return await self.model.with_structured_output(schema).ainvoke(prompt, config=self._config)
                    except Exception:
                        return _parse((await self.model.ainvoke(prompt, config=self._config)).content, schema)
                except Exception:
                    if attempt == ATTEMPTS - 1:
                        raise
                    await asyncio.sleep(_backoff(attempt))
