import os
import random
from typing import List, Optional, Literal, Dict, Any

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None  # type: ignore

try:
    from litellm import acompletion
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "litellm must be installed. Add it to requirements.txt"
    ) from exc


Side = Literal["affirmative", "negative"]


DEFAULT_CANDIDATE_MODELS: List[str] = [
    # Works if OPENAI_API_KEY is set
    "gpt-4o-mini",
    # Works if ANTHROPIC_API_KEY is set
    "claude-3-haiku-20240307",
    # Works if GEMINI_API_KEY is set
    "gemini/gemini-1.5-flash",
    # Works if XAI_API_KEY is set
    "grok-2",
]


class DebateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    models: Optional[List[str]] = None
    turns: int = Field(1, ge=1, le=5, description="Number of turns per side")
    use_search: bool = Field(True, description="Use Tavily search for research context")


class DebateTurn(BaseModel):
    question: str
    turn: int
    side: Side
    model: str
    message: str


app = FastAPI(title="Debate API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {"status": "ok"}


def _select_two_models(preferred_models: Optional[List[str]]) -> List[str]:
    candidates_env = os.getenv("DEBATE_MODELS")
    if candidates_env:
        candidates = [m.strip() for m in candidates_env.split(",") if m.strip()]
    else:
        candidates = DEFAULT_CANDIDATE_MODELS.copy()

    # If request provides models, use them if at least two
    if preferred_models:
        unique = [m for i, m in enumerate(preferred_models) if m and m not in preferred_models[:i]]
        if len(unique) >= 2:
            return unique[:2]

    # Fallback: choose randomly
    if len(candidates) >= 2:
        return random.sample(candidates, 2)
    if len(candidates) == 1:
        return [candidates[0], candidates[0]]
    # Ultimate fallback: two copies of a common model name
    return ["gpt-4o-mini", "claude-3-haiku-20240307"]


def _assign_sides() -> Dict[str, Side]:
    sides: List[Side] = ["affirmative", "negative"]
    random.shuffle(sides)
    return {"speaker_a": sides[0], "speaker_b": sides[1]}


def _build_research_context(question: str, use_search: bool) -> str:
    if not use_search:
        return ""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or TavilyClient is None:
        return ""
    try:
        client = TavilyClient(api_key=api_key)
        result = client.search(query=question, max_results=5, search_depth="advanced")
        findings = result.get("results", []) if isinstance(result, dict) else []
        if not findings:
            return ""
        snippets: List[str] = []
        for item in findings[:5]:
            title = item.get("title") or ""
            url = item.get("url") or ""
            content = item.get("content") or ""
            if not content:
                continue
            trimmed = content.strip().replace("\n", " ")
            if len(trimmed) > 220:
                trimmed = trimmed[:220].rstrip() + "…"
            snippet = f"- {title}: {trimmed} (source: {url})"
            snippets.append(snippet)
        return "\n".join(snippets)
    except Exception:
        return ""


def _system_prompt(side: Side, question: str, research_context: str) -> str:
    base = (
        "You are participating in a structured debate.\n"
        f"Your assigned side: {side.upper()}.\n"
        f"Debate question: {question}\n\n"
        "Rules:\n"
        "- Review the full conversation before replying.\n"
        "- Do NOT repeat prior arguments; extend or counter with new points.\n"
        "- Be concise, persuasive, and factual.\n"
        "- Reply in 280 characters or fewer.\n"
        "- If citing, do brief inline references like [source].\n"
    )
    if research_context:
        base += "\nResearch context (summarized):\n" + research_context
    return base


async def _model_reply(
    model_name: str,
    side: Side,
    question: str,
    history_messages: List[Dict[str, str]],
    research_context: str,
    max_characters: int = 280,
) -> str:
    system = _system_prompt(side=side, question=question, research_context=research_context)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": question})
    messages.extend(history_messages)

    # Approximate 280 chars with tokens. ~4 chars/token → ~70 tokens
    max_tokens = int(max_characters / 4) or 70

    response = await acompletion(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    content = (
        (response.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if len(content) > max_characters:
        content = content[:max_characters].rstrip()
    return content


@app.post("/debate", response_model=List[DebateTurn])
async def debate(req: DebateRequest) -> List[DebateTurn]:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    model_a, model_b = _select_two_models(req.models)
    sides_map = _assign_sides()  # speaker_a / speaker_b → side
    research_context = _build_research_context(req.question.strip(), req.use_search)

    # Conversation history shared across both models
    history_messages: List[Dict[str, str]] = []
    transcript: List[DebateTurn] = []

    for turn_index in range(1, req.turns + 1):
        # Speaker A
        side_a: Side = sides_map["speaker_a"]
        reply_a = await _model_reply(
            model_name=model_a,
            side=side_a,
            question=req.question.strip(),
            history_messages=history_messages,
            research_context=research_context,
        )
        history_messages.append({"role": "assistant", "content": reply_a})
        transcript.append(
            DebateTurn(
                question=req.question.strip(),
                turn=turn_index,
                side=side_a,
                model=model_a,
                message=reply_a,
            )
        )

        # Speaker B (sees A's reply in history)
        side_b: Side = sides_map["speaker_b"]
        reply_b = await _model_reply(
            model_name=model_b,
            side=side_b,
            question=req.question.strip(),
            history_messages=history_messages,
            research_context=research_context,
        )
        history_messages.append({"role": "assistant", "content": reply_b})
        transcript.append(
            DebateTurn(
                question=req.question.strip(),
                turn=turn_index,
                side=side_b,
                model=model_b,
                message=reply_b,
            )
        )

    return transcript

