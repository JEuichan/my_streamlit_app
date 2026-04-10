"""Seed-aligned LangGraph workflow: ReAct + Serper + stubs, evaluate/retry/human gate."""

from __future__ import annotations

import operator
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, Literal, Sequence, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from daily_assistant_core import build_llm

load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

CONFIDENCE_OK = 0.65
MAX_AUTO_RETRIES = 2


def _ai_text_content(msg: AIMessage | None) -> str:
    """Normalize AIMessage.content (str or list of blocks) to plain text."""
    if msg is None:
        return ""
    c = msg.content
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts).strip()
    return str(c).strip() if c else ""


class EvalResult(BaseModel):
    """Structured evaluation of draft answer quality."""

    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in [0,1]")
    sufficient_sources: bool = Field(description="Whether cited/snippet evidence is enough")
    rationale: str = Field(max_length=280, description="One short sentence")


class ResearchState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    search_results: list[dict[str, str]]
    draft_answer: str | None
    confidence: float | None
    sufficient_sources: bool | None
    retry_count: int
    branch_reasons: Annotated[list[str], operator.add]
    awaiting_user: bool
    user_decision: Literal["continue", "stop"] | None
    final_answer: str | None


def _serper_fetch(query: str) -> tuple[str, list[dict[str, str]]]:
    key = os.getenv("SERPER_API_KEY")
    hits: list[dict[str, str]] = []
    if not key:
        return (
            "SERPER_API_KEY가 없습니다. .env에 SERPER_API_KEY를 설정하세요.",
            hits,
        )
    try:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": 8},
            timeout=45.0,
        )
        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic") or []
        lines: list[str] = []
        for i, item in enumerate(organic, 1):
            title = (item.get("title") or "")[:200]
            link = item.get("link") or ""
            snippet = (item.get("snippet") or "")[:400]
            hits.append({"title": title, "url": link, "snippet": snippet})
            lines.append(f"{i}. {title}\n   URL: {link}\n   {snippet}")
        if not lines:
            return ("검색 결과가 비었습니다. 쿼리를 바꿔 다시 시도하세요.", hits)
        return ("\n\n".join(lines), hits)
    except Exception as e:
        return (f"Serper 요청 실패: {e}", hits)


def gate(state: ResearchState) -> Literal["finalize_stop", "resume_prep", "react"]:
    if state.get("user_decision") == "stop":
        return "finalize_stop"
    if state.get("awaiting_user") and state.get("user_decision") == "continue":
        return "resume_prep"
    return "react"


def react_node(state: ResearchState) -> dict:
    """Single ReAct loop: model + tools until no tool calls."""
    collected: list[dict[str, str]] = []

    @tool
    def serper_web_search(query: str) -> str:
        """Search the public web via Serper (Google). Use for facts, dates, and citations."""
        text, hits = _serper_fetch(query)
        collected.extend(hits)
        return text

    @tool
    def stub_register_claim(claim: str) -> str:
        """Stub: pretend to register a factual claim for an internal checklist (portfolio demo)."""
        return f"[stub] claim registered: {claim[:120]}"

    @tool
    def stub_word_stats(text: str) -> str:
        """Stub: return simple length stats for a snippet (portfolio demo)."""
        words = len(text.split())
        return f"[stub] words={words}, chars={len(text)}"

    tools = [serper_web_search, stub_register_claim, stub_word_stats]
    tools_by_name = {t.name: t for t in tools}
    llm = build_llm().bind_tools(tools)

    sys = SystemMessage(
        content=(
            "You are a careful research assistant. "
            "Use serper_web_search for up-to-date or factual web information. "
            "Call stub tools only if they help demonstrate the workflow. "
            "Answer in Korean when the user writes in Korean. "
            "Cite sources implicitly in prose (titles/URLs appear in tool output)."
        )
    )
    base = list(state.get("messages") or [])
    head = base[:4]
    has_system = any(isinstance(m, SystemMessage) for m in head)
    messages = base if has_system else [sys] + base

    new_msgs: list[BaseMessage] = []
    max_rounds = 14
    for _ in range(max_rounds):
        ai: AIMessage = llm.invoke(messages + new_msgs)
        new_msgs.append(ai)
        if not getattr(ai, "tool_calls", None):
            break
        for tc in ai.tool_calls:
            name = tc.get("name") or ""
            args = tc.get("args") or {}
            tid = tc.get("id") or ""
            fn = tools_by_name.get(name)
            if fn is None:
                out = f"unknown tool: {name}"
            else:
                try:
                    out = fn.invoke(args)
                except Exception as e:
                    out = f"tool error: {e}"
            new_msgs.append(ToolMessage(content=str(out), tool_call_id=tid, name=name))

    last_ai = next((m for m in reversed(new_msgs) if isinstance(m, AIMessage)), None)
    draft = _ai_text_content(last_ai) if last_ai else ""

    return {
        "messages": new_msgs,
        "search_results": collected,
        "draft_answer": draft,
    }


def evaluate_node(state: ResearchState) -> dict:
    q = state.get("user_query") or ""
    draft = state.get("draft_answer") or ""
    n_src = len(state.get("search_results") or [])
    evaluator = build_llm().with_structured_output(EvalResult)
    er = evaluator.invoke(
        [
            SystemMessage(
                content=(
                    "Evaluate whether the draft answer is well-supported and confident enough "
                    "for a light fact-check. Consider number of search hits and whether the answer "
                    "addresses the question. Be slightly strict for demo branching."
                )
            ),
            HumanMessage(
                content=f"Question:\n{q}\n\nDraft answer:\n{draft}\n\nSearch result rows: {n_src}"
            ),
        ]
    )
    return {
        "confidence": er.confidence,
        "sufficient_sources": er.sufficient_sources,
    }


def route_after_eval(state: ResearchState) -> Literal["finalize", "retry", "human"]:
    conf = float(state.get("confidence") or 0.0)
    suff = bool(state.get("sufficient_sources"))
    if conf >= CONFIDENCE_OK and suff:
        return "finalize"
    rc = int(state.get("retry_count") or 0)
    if rc < MAX_AUTO_RETRIES:
        return "retry"
    return "human"


def retry_prep_node(state: ResearchState) -> dict:
    rc = int(state.get("retry_count") or 0) + 1
    reason = (
        f"신뢰도·출처 기준 미충족 → 자동 재시도 ({rc}/{MAX_AUTO_RETRIES})"
    )
    return {
        "retry_count": rc,
        "branch_reasons": [reason],
        "messages": [
            SystemMessage(
                content=(
                    "이전 답변은 검증 단계에서 불충분했습니다. "
                    "추가 검색을 하고, 가능하면 출처가 드러나게 다시 답하세요."
                )
            )
        ],
    }


def human_pause_node(state: ResearchState) -> dict:
    return {
        "awaiting_user": True,
        "branch_reasons": [
            "자동 재시도 한도 도달 → 사용자 확인 필요 (계속/중단)"
        ],
    }


def finalize_node(state: ResearchState) -> dict:
    text = (state.get("draft_answer") or "").strip()
    sources = state.get("search_results") or []
    parts = [text if text else "(답변 없음)", "", "### 출처", ""]
    if not sources:
        parts.append("- (웹 검색 결과 없음 또는 Serper 미설정)")
    else:
        for i, s in enumerate(sources[:8], 1):
            title = s.get("title") or ""
            url = s.get("url") or ""
            snip = (s.get("snippet") or "")[:160]
            parts.append(f"{i}. [{title}]({url}) — {snip}")
    return {
        "final_answer": "\n".join(parts),
        "awaiting_user": False,
        "branch_reasons": ["검증 통과 → 최종 답변 및 출처 정리"],
    }


def finalize_stop_node(state: ResearchState) -> dict:
    return {
        "final_answer": "사용자가 추가 조사를 중단했습니다.",
        "awaiting_user": False,
    }


def resume_prep_node(state: ResearchState) -> dict:
    return {
        "awaiting_user": False,
        "user_decision": None,
        "branch_reasons": ["사용자가 추가 조사를 선택함 → ReAct 재실행"],
        "messages": [
            SystemMessage(
                content="사용자가 추가 검색을 승인했습니다. 더 많은 출처를 찾아 답을 개선하세요."
            )
        ],
    }


def build_research_graph():
    g = StateGraph(ResearchState)
    g.add_node("react", react_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("retry_prep", retry_prep_node)
    g.add_node("human_pause", human_pause_node)
    g.add_node("finalize", finalize_node)
    g.add_node("finalize_stop", finalize_stop_node)
    g.add_node("resume_prep", resume_prep_node)

    g.add_conditional_edges(
        START,
        gate,
        {
            "finalize_stop": "finalize_stop",
            "resume_prep": "resume_prep",
            "react": "react",
        },
    )
    g.add_edge("resume_prep", "react")
    g.add_edge("react", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        route_after_eval,
        {"finalize": "finalize", "retry": "retry_prep", "human": "human_pause"},
    )
    g.add_edge("retry_prep", "react")
    g.add_edge("finalize", END)
    g.add_edge("finalize_stop", END)
    g.add_edge("human_pause", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


def new_thread_config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}
