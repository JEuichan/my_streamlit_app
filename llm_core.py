"""OpenAI Chat (gpt-5-mini) — 단일 레포 배포용 최소 모듈."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def build_llm() -> ChatOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다.")
    return ChatOpenAI(model="gpt-5-mini", api_key=key)
