import uuid
import asyncio

from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from string import Template

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from litellm import acompletion
import json
import yaml
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load API keys from .env into environment
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
FORUMS_DIR: Path = DATA_DIR
FORUMS_DIR.mkdir(parents=True, exist_ok=True)

def _load_app_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(f"App config not found at '{config_path}'")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("App config must be a mapping at the top-level")
    return data

APP_CONFIG: Dict[str, Any] = _load_app_config()

max_characters: int = int(APP_CONFIG["max_characters"])
max_completion_tokens: int = int(APP_CONFIG["max_completion_tokens"])
max_input_tokens: int = int(APP_CONFIG.get("max_input_tokens", 64))
max_reasoning_tokens: int = int(APP_CONFIG.get("max_reasoning_tokens", 10000))

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

class ForumPost(BaseModel):
    model: str
    message: str

class ForumStatus(str, Enum):
    pending = "pending"
    complete = "complete"
    error = "error"

class Forum(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    id: str
    question: str
    models: List[str]
    transcript: List[ForumPost]
    status: ForumStatus = Field(default=ForumStatus.pending)

app = FastAPI(title="Forum API", version="0.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper functions ---
def _validate_model_enabled(model_id: str) -> None:
    enabled_models = set(_load_models_from_yaml())
    mid = (model_id or "").strip()
    if mid not in enabled_models:
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
) -> str:
    system = _system_prompt(question=question)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": question})

    response = await acompletion(
        model=model_name,
        messages=messages,
        max_tokens=max_completion_tokens,
    )
    def _extract_content(resp: Dict[str, Any]) -> str:
        try:
            content_val = resp["choices"][0]["message"]["content"]
            return content_val.strip() if isinstance(content_val, str) else ""
        except Exception:
            return ""

    content = _extract_content(response)
    if len(content) > max_characters:
        content = content[:max_characters].rstrip()
    return content or "[empty]"

# In-memory store for forums
FORUMS: Dict[str, Forum] = {}
FORUMS_LOCK = asyncio.Lock()

# --- Persistence helpers ---
def _forum_path(forum_id: str) -> Path:
    return FORUMS_DIR / f"{forum_id}.json"


def _save_forum_to_disk(forum: Forum) -> None:
    try:
        path = _forum_path(forum.id)
        data = forum.model_dump(mode="json")
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _load_forum_from_disk(forum_id: str) -> Optional[Forum]:
    try:
        path = _forum_path(forum_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Forum(**data)
    except Exception:
        return None

# --- Endpoints ---
async def _generate_forum_transcript(
    forum_id: str,
    question: str,
    model_names: List[str],
) -> None:
    transcript: List[ForumPost] = []
    try:
        # Run all model calls concurrently
        tasks = [
            _model_reply(model_name=m, question=question)
            for m in model_names
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for model_name, result in zip(model_names, results):
            if isinstance(result, Exception):
                message = f"[error] {type(result).__name__}: {str(result)}"
            else:
                message = result or ""
            transcript.append(ForumPost(model=model_name, message=message))

        async with FORUMS_LOCK:
            forum = FORUMS.get(forum_id)
            if forum is not None:
                forum.transcript = transcript
                forum.status = ForumStatus.complete
                _save_forum_to_disk(forum)
    except Exception as e:
        async with FORUMS_LOCK:
            forum = FORUMS.get(forum_id)
            if forum is not None:
                forum.transcript = transcript
                forum.status = ForumStatus.error
                _save_forum_to_disk(forum)

@app.post("/forum", response_model=Forum)
async def create_forum(
    req: ForumCreateRequest,
) -> Forum:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    # Validate requested models exist and are enabled
    enabled_models_list = _load_models_from_yaml()
    max_selectable = len(enabled_models_list)

    # Pre-check: ensure requested unique count does not exceed available models
    requested_unique = set()
    for m in req.models:
        if isinstance(m, str) and m.strip():
            requested_unique.add(m.strip())
    if max_selectable == 0:
        raise HTTPException(status_code=400, detail="No models are available to select")
    if len(requested_unique) > max_selectable:
        raise HTTPException(
            status_code=400,
            detail=f"You can select up to {max_selectable} models",
        )

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

    # Independent responses: show placeholders immediately
    transcript: List[ForumPost] = [
        ForumPost(model=m, message="[pending]") for m in unique_models
    ]

    # Enforce small input budget
    bounded_question = _truncate_question_to_input_budget(req.question)

    forum_id = str(uuid.uuid4())
    forum = Forum(
        id=forum_id,
        question=bounded_question,
        models=unique_models,
        transcript=transcript,
        status=ForumStatus.pending,
    )

    async with FORUMS_LOCK:
        FORUMS[forum_id] = forum
        _save_forum_to_disk(forum)

    # Kick off background generation and return immediately
    asyncio.create_task(
        _generate_forum_transcript(
            forum_id=forum_id,
            question=bounded_question,
            model_names=unique_models,
        )
    )
    return forum

@app.get("/forum/{forum_id}", response_model=Forum)
async def get_forum(forum_id: str) -> Forum:
    forum = FORUMS.get(forum_id)
    if forum is None:
        loaded = _load_forum_from_disk(forum_id)
        if loaded is not None:
            async with FORUMS_LOCK:
                FORUMS[forum_id] = loaded
            return loaded
        raise HTTPException(status_code=404, detail="Forum not found")
    return forum

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}