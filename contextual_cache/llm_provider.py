"""
LLM provider abstraction with circuit-breaker protection and retry logic.

Supports Ollama (local), Groq, and OpenAI backends.
All calls are async via httpx.  Cost tracking per call.
Retries transient failures with exponential backoff + jitter.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Optional

import httpx

from .circuit_breaker import CircuitBreaker
from .config import LLMBackend, settings

logger = logging.getLogger(__name__)

# HTTP status codes that are safe to retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMProvider:
    """
    Unified async LLM abstraction.

    - Ollama:  POST /api/generate  (local, no key needed)
    - Groq:    POST /openai/v1/chat/completions  (API key required)
    - OpenAI:  POST /v1/chat/completions  (API key required)

    Features:
    - Exponential backoff with jitter on transient failures
    - Respects Retry-After header on 429 responses
    - Structured timeouts (connect vs read)
    - API key validation for cloud backends
    """

    # Rough cost estimates per 1K output tokens (USD)
    COST_PER_1K_TOKENS = {
        LLMBackend.OLLAMA: 0.0,      # free, local
        LLMBackend.GROQ: 0.00027,    # llama-3 on Groq (approx)
        LLMBackend.OPENAI: 0.015,    # gpt-4o-mini (approx)
    }

    def __init__(
        self,
        backend: LLMBackend = settings.llm_backend,
        model: str = settings.llm_model,
        base_url: str = settings.llm_base_url,
        api_key: Optional[str] = settings.llm_api_key,
        timeout_s: float = settings.llm_timeout_s,
        max_tokens: int = settings.llm_max_tokens,
        max_retries: int = settings.llm_max_retries,
        retry_base_delay_s: float = settings.llm_retry_base_delay_s,
    ) -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay_s

        # Validate API key for cloud backends
        if backend in (LLMBackend.GROQ, LLMBackend.OPENAI) and not api_key:
            logger.warning(
                "No API key set for %s backend. Requests will likely fail.",
                backend.value,
            )

        # Structured timeout: separate connect and read
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.llm_connect_timeout_s,
                read=settings.llm_read_timeout_s,
                write=30.0,
                pool=10.0,
            )
        )
        self._circuit = CircuitBreaker(name=f"llm-{backend.value}")

        # Stats
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.total_latency_ms = 0.0
        self.total_retries = 0

    async def generate(self, prompt: str,
                       system_prompt: Optional[str] = None,
                       chat_history: Optional[list] = None) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The current user query.
            system_prompt: Optional system instruction.
            chat_history: Optional list of {"role": "user"|"assistant", "content": str}
                          for multi-turn context. Most recent messages last.

        Returns an LLMResponse with text, token count, cost, and latency.
        Raises on circuit-breaker rejection or LLM error after retries.
        """
        t0 = time.monotonic()

        async def _call():
            if self.backend == LLMBackend.OLLAMA:
                if chat_history:
                    return await self._call_with_retry(
                        self._call_ollama_chat, prompt, system_prompt,
                        chat_history=chat_history,
                    )
                return await self._call_with_retry(
                    self._call_ollama, prompt, system_prompt
                )
            else:
                return await self._call_with_retry(
                    self._call_openai_compat, prompt, system_prompt,
                    chat_history=chat_history,
                )

        result = await self._circuit.call(_call)
        latency_ms = (time.monotonic() - t0) * 1000

        self.total_calls += 1
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.total_tokens += result.input_tokens + result.output_tokens
        self.total_cost_usd += result.cost_usd
        self.total_latency_ms += latency_ms
        result.latency_ms = latency_ms

        return result

    async def _call_with_retry(self, fn, prompt: str,
                                system_prompt: Optional[str],
                                chat_history: Optional[list] = None) -> LLMResponse:
        """Retry transient failures with exponential backoff + jitter."""
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                if chat_history is not None:
                    return await fn(prompt, system_prompt, chat_history=chat_history)
                return await fn(prompt, system_prompt)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status not in _RETRYABLE_STATUS_CODES or attempt == self._max_retries:
                    raise
                last_exc = e
                delay = self._backoff_delay(attempt, e.response)
                logger.warning(
                    "LLM %s returned %d, retrying in %.1fs (attempt %d/%d)",
                    self.backend.value, status, delay, attempt + 1, self._max_retries,
                )
                self.total_retries += 1
                await asyncio.sleep(delay)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
                    httpx.PoolTimeout) as e:
                if attempt == self._max_retries:
                    raise
                last_exc = e
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM %s connection error: %s, retrying in %.1fs (attempt %d/%d)",
                    self.backend.value, type(e).__name__, delay,
                    attempt + 1, self._max_retries,
                )
                self.total_retries += 1
                await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    def _backoff_delay(self, attempt: int,
                       response: Optional[httpx.Response] = None) -> float:
        """Exponential backoff with jitter. Respects Retry-After header."""
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        # Exponential backoff: base * 2^attempt + jitter
        delay = self._retry_base_delay * (2 ** attempt)
        jitter = random.uniform(0, delay * 0.25)
        return min(delay + jitter, 60.0)  # cap at 60s

    async def _call_ollama(self, prompt: str,
                           system_prompt: Optional[str]) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = await self._client.post(
            f"{self.base_url}/api/generate", json=payload
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Ollama returned 404 Not Found. Please ensure you have "
                    f"pulled the model '{self.model}' using "
                    f"`ollama pull {self.model}`."
                ) from e
            raise

        data = resp.json()

        text = data.get("response", "")
        eval_count = data.get("eval_count", len(text.split()))
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return LLMResponse(
            text=text,
            output_tokens=eval_count,
            input_tokens=prompt_eval_count,
            cost_usd=0.0,
        )

    async def _call_ollama_chat(self, prompt: str,
                                system_prompt: Optional[str],
                                chat_history: Optional[list] = None) -> LLMResponse:
        """Ollama /api/chat endpoint for multi-turn conversations."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": self.max_tokens},
        }

        resp = await self._client.post(
            f"{self.base_url}/api/chat", json=payload
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Ollama returned 404 Not Found. Please ensure you have "
                    f"pulled the model '{self.model}' using "
                    f"`ollama pull {self.model}`."
                ) from e
            raise

        data = resp.json()
        text = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", len(text.split()))
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return LLMResponse(
            text=text,
            output_tokens=eval_count,
            input_tokens=prompt_eval_count,
            cost_usd=0.0,
        )

    async def _call_openai_compat(self, prompt: str,
                                   system_prompt: Optional[str],
                                   chat_history: Optional[list] = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": prompt})

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/v1/chat/completions"
        if self.backend == LLMBackend.GROQ:
            url = f"{self.base_url}/openai/v1/chat/completions"

        resp = await self._client.post(
            url,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("LLM returned empty choices list.")
        choice = choices[0]
        text = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        output_tokens = usage.get("completion_tokens", len(text.split()))
        input_tokens = usage.get("prompt_tokens", 0)

        cost_per_1k = self.COST_PER_1K_TOKENS.get(self.backend, 0.01)
        cost = (output_tokens / 1000) * cost_per_1k

        return LLMResponse(
            text=text,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            cost_usd=cost,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def get_stats(self) -> dict:
        return {
            "backend": self.backend.value,
            "model": self.model,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(
                self.total_latency_ms / max(self.total_calls, 1), 2
            ),
            "total_retries": self.total_retries,
            "circuit_state": self._circuit.state.value,
        }


class LLMResponse:
    """Response from an LLM call."""

    __slots__ = ("text", "input_tokens", "output_tokens", "cost_usd", "latency_ms")

    def __init__(self, text: str, output_tokens: int, cost_usd: float,
                 latency_ms: float = 0.0, input_tokens: int = 0) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.latency_ms = latency_ms
