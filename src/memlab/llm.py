"""LLM 프로바이더 계층 — 메소드들이 공유하는 범용 인프라.

모든 메모리 메소드(MemoryOS 재구현, 미래의 변형)와 답변 생성 모듈은
LLMProvider 인터페이스만 알고, 실제 공급자(Groq 등)는 상속으로 구현한다.
호출 수·토큰 집계는 베이스가 공통 제공 — 실험 비용 측정에 쓴다.

- chat: 자유 텍스트. 채점되는 출력(답변 생성)은 이쪽 —
  형식 제약이 답 스타일을 바꾸면 F1이 왜곡되므로.
- chat_model: Pydantic 모델로 스키마 강제 + 검증. 내부 북키핑
  호출(판정·추출)용. 검증 실패 시 에러를 모델에게 되먹여 1회 재요청(reask).
- 429는 서버의 retry-after 헤더를 존중하고, 없을 때만 지수 백오프.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from memlab import config
from memlab.config import groq_api_key

TModel = TypeVar("TModel", bound=BaseModel)

# 서버가 이보다 긴 대기를 지시하면 조용히 자는 대신 실패시킨다 (침묵 방지)
MAX_RETRY_WAIT_SECONDS = 900.0


class LLMProvider(ABC):
    def __init__(self):
        self.calls = 0
        self.total_tokens = 0

    @abstractmethod
    def chat(
        self, system: str, user: str, *, temperature: float = 0.7, max_tokens: int = 2000
    ) -> str:
        """system+user 메시지 1회 호출 → 응답 텍스트."""

    @abstractmethod
    def chat_model(
        self,
        system: str,
        user: str,
        response_model: type[TModel],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> TModel:
        """스키마(Pydantic 모델)를 강제한 1회 호출 → 검증된 모델 인스턴스."""


class OpenAICompatProvider(LLMProvider):
    """OpenAI 호환 엔드포인트 공용 구현 — Groq, Ollama 등이 상속한다."""

    def __init__(self, model: str, base_url: str, api_key: str,
                 max_retries: int = 5, timeout: float = 60.0):
        super().__init__()
        # timeout: SDK 기본값(600초)은 연결이 멈추면 10분을 조용히 기다린다.
        # max_retries=0: 재시도는 우리 루프가 담당 (SDK 내부 재시도와 중복 방지)
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0,
        )
        self.model = model
        self.max_retries = max_retries

    def chat(
        self, system: str, user: str, *, temperature: float = 0.7, max_tokens: int = 2000
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._complete(messages, temperature=temperature, max_tokens=max_tokens)

    def chat_model(
        self,
        system: str,
        user: str,
        response_model: type[TModel],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> TModel:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
            },
        }
        raw = ""
        for _ in range(2):  # 최초 1회 + 검증 실패 시 reask 1회
            raw = self._complete_structured(
                messages, response_format, temperature=temperature, max_tokens=max_tokens
            )
            try:
                return response_model.model_validate_json(raw)
            except ValidationError as error:
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response failed validation:\n{error}\n"
                            "Respond again with ONLY valid JSON matching the schema."
                        ),
                    },
                ]
        raise ValueError(
            f"{response_model.__name__} 검증이 reask 후에도 실패: {raw[:200]!r}"
        )

    # ── 내부 ─────────────────────────────────────────────────────

    def _complete_structured(
        self, messages: list[dict], response_format: dict, *, temperature, max_tokens
    ) -> str:
        try:
            return self._complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except BadRequestError:  # json_schema 미지원 모델 → json_object 폴백
            return self._complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

    def _complete(
        self, messages: list[dict], *, temperature, max_tokens, response_format=None
    ) -> str:
        """1회 완성 호출 — 429 재시도와 사용량 집계를 담당."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self.calls += 1
                self.total_tokens += response.usage.total_tokens
                return (response.choices[0].message.content or "").strip()
            except RateLimitError as error:
                if attempt == self.max_retries - 1:
                    raise
                delay = self._retry_delay(error, attempt)
                if delay > MAX_RETRY_WAIT_SECONDS:
                    raise
                if delay > 5:
                    print(f"    [rate-limit] {delay:.0f}초 대기")
                time.sleep(delay)
            except (APITimeoutError, APIConnectionError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(float(2**attempt))
        raise RuntimeError("unreachable")

    @staticmethod
    def _retry_delay(error: APIStatusError, attempt: int) -> float:
        """서버가 알려주는 대기시간(retry-after)을 존중, 없으면 지수 백오프."""
        retry_after = error.response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.1)
            except ValueError:
                pass
        return float(2**attempt)


class GroqProvider(OpenAICompatProvider):
    """Groq free API. 주의: 무료 티어는 TPM 벽이 좁다 (2026-07-04 실측)."""

    def __init__(self, model: str | None = None, max_retries: int = 5):
        super().__init__(
            model=model or config.GROQ_MODEL,
            base_url=config.GROQ_BASE_URL,
            api_key=groq_api_key(),
            max_retries=max_retries,
        )


class LMStudioProvider(OpenAICompatProvider):
    """로컬 LM Studio 서버(MLX 엔진) — rate limit 없음.
    로컬 추론이라 타임아웃은 넉넉하게."""

    def __init__(self, model: str | None = None, max_retries: int = 3):
        super().__init__(
            model=model or config.LMSTUDIO_MODEL,
            base_url=config.LMSTUDIO_BASE_URL,
            api_key="lm-studio",  # LM Studio는 키를 검사하지 않지만 SDK가 요구
            max_retries=max_retries,
            timeout=300.0,
        )


def default_provider() -> LLMProvider:
    """config.LLM_PROVIDER에 따른 기본 프로바이더."""
    if config.LLM_PROVIDER == "lmstudio":
        return LMStudioProvider()
    return GroqProvider()
