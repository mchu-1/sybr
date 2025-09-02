import uuid

from typing import List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from string import Template

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from litellm import acompletion
import yaml
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load API keys from .env into environment
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

def _load_app_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(f"App config not found at '{config_path}'")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("App config must be a mapping at the top-level")
    return data

APP_CONFIG: Dict[str, Any] = _load_app_config()

default_turns: int = int(APP_CONFIG["default_turns"])
max_characters: int = int(APP_CONFIG["max_characters"])
max_completion_tokens: int = int(APP_CONFIG["max_completion_tokens"])
max_input_tokens: int = int(APP_CONFIG.get("max_input_tokens", 64))
max_reasoning_tokens: int = int(APP_CONFIG.get("max_reasoning_tokens", 10000))
temperature: float = float(APP_CONFIG["temperature"])
top_p: float = float(APP_CONFIG["top_p"])

def _get_system_template() -> Template:
    """Load system prompt template."""
    template_text = (PROJECT_ROOT / "system.md").read_text(encoding="utf-8")
    return Template(template_text)

def _load_models_from_yaml() -> List[str]:
    """Load enabled model IDs."""
    try:
        config_path = PROJECT_ROOT / "models.yaml"
        if not config_path.exists():
            return []
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        items = data.get("models", []) if isinstance(data, dict) else []
        model_ids: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            enabled = item.get("enabled", True)
            model_id = item.get("id")
            if enabled and isinstance(model_id, str) and model_id.strip():
                model_ids.append(model_id.strip())
        return model_ids
    except Exception:
        return []

# --- Data models ---
class ForumCreateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    models: List[str] = Field(..., min_items=1, description="List of model IDs to participate")
    turns: int = Field(
        default_turns,
        ge=1,
        description="Number of messages per model",
    )

class ForumPost(BaseModel):
    turn: int
    model: str
    message: str

class ForumThread(BaseModel):
    id: str
    question: str
    models: List[str]
    transcript: List[ForumPost]

app = FastAPI(title="Forum API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper functions ---
def _validate_model_enabled(model_id: str) -> None:
    enabled_models = set(_load_models_from_yaml())
    if model_id not in enabled_models:
        raise HTTPException(status_code=400, detail=(
            f"Model '{model_id}' is not enabled or not found in models.yaml."
        ))

def _system_prompt(question: str) -> str:
    template = _get_system_template()
    return template.safe_substitute(
        QUESTION=question,
        MAX_CHARACTERS=str(max_characters),
    )

def _truncate_question_to_input_budget(question: str) -> str:
    # Approximate tokens as chars/4 and limit accordingly
    approx_chars_per_token = 4
    char_budget = max(1, max_input_tokens * approx_chars_per_token)
    q = (question or "").strip()
    if len(q) <= char_budget:
        return q
    return q[:char_budget].rstrip()

async def _model_reply(
    model_name: str,
    question: str,
    history_messages: List[Dict[str, str]],
) -> str:
    system = _system_prompt(question=question)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": question})
    messages.extend(history_messages)

    # Best-effort reasoning token support via passthrough
    extra_body = {
        "reasoning": {
            "effort": "medium",
            "tokens": max_reasoning_tokens,
            "budget_tokens": max_reasoning_tokens,
        }
    }

    response = await acompletion(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
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

# In-memory store for forum threads
FORUM_THREADS: Dict[str, ForumThread] = {}

# --- Endpoints ---
@app.post("/forums", response_model=ForumThread)
async def create_forum_thread(req: ForumCreateRequest) -> ForumThread:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    # Validate requested models exist and are enabled
    unique_models = []
    seen = set()
    for m in req.models:
        if not isinstance(m, str) or not m.strip():
            continue
        mid = m.strip()
        if mid in seen:
            continue
        _validate_model_enabled(mid)
        seen.add(mid)
        unique_models.append(mid)
    if not unique_models:
        raise HTTPException(status_code=400, detail="No valid models provided")

    # Shared conversation history across all models
    history_messages: List[Dict[str, str]] = []
    transcript: List[ForumPost] = []

    # Enforce small input budget
    bounded_question = _truncate_question_to_input_budget(req.question)

    # Round-robin k turns per model
    for turn_index in range(1, req.turns + 1):
        for model_name in unique_models:
            reply = await _model_reply(
                model_name=model_name,
                question=bounded_question,
                history_messages=history_messages,
            )
            history_messages.append({"role": "assistant", "content": reply})
            transcript.append(
                ForumPost(
                    turn=turn_index,
                    model=model_name,
                    message=reply,
                )
            )

    thread_id = str(uuid.uuid4())
    thread = ForumThread(
        id=thread_id,
        question=bounded_question,
        models=unique_models,
        transcript=transcript,
    )

    FORUM_THREADS[thread_id] = thread
    return thread

@app.get("/forums/{thread_id}", response_model=ForumThread)
async def get_forum_thread(thread_id: str) -> ForumThread:
    thread = FORUM_THREADS.get(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Forum thread not found")
    return thread

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}