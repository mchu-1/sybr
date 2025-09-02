import uuid
import asyncio

from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from string import Template
from datetime import datetime, timezone
from html import escape as html_escape

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from litellm import acompletion
import json
import yaml
from dotenv import load_dotenv
from urllib.parse import quote as url_quote

# --- Configuration ---
load_dotenv()  # Load API keys from .env into environment
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
APP_DIR: Path = Path(__file__).resolve().parent
STATIC_DIR: Path = APP_DIR / "static"
TEMPLATES_DIR: Path = APP_DIR / "templates"
DATA_DIR: Path = PROJECT_ROOT / "data"
FORUMS_DIR: Path = DATA_DIR
FORUMS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_JSONL: Path = DATA_DIR / "conversations.jsonl"

def _load_app_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(f"App config not found at '{config_path}'")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("App config must be a mapping at the top-level")
    return data

APP_CONFIG: Dict[str, Any] = _load_app_config()

max_tokens: int = int(APP_CONFIG["max_tokens"])
max_chars: int = int(APP_CONFIG["max_chars"])
max_input: int = int(APP_CONFIG.get("max_input"))
max_time: int = int(APP_CONFIG.get("max_time"))
approx_chars_per_token: int = 4

def _get_system_template() -> Template:
    """Load system prompt template."""
    template_text = (PROJECT_ROOT / "system.md").read_text(encoding="utf-8")
    return Template(template_text)

def _ensure_conversations_file() -> None:
    try:
        if not CONVERSATIONS_JSONL.exists():
            CONVERSATIONS_JSONL.touch()
    except Exception:
        pass

def _append_conversation_jsonl(record: Dict[str, Any]) -> None:
    try:
        with CONVERSATIONS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _load_conversation_ids() -> set:
    ids: set = set()
    if not CONVERSATIONS_JSONL.exists():
        return ids
    try:
        with CONVERSATIONS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj.get("id")
                    if isinstance(cid, str):
                        ids.add(cid)
                except Exception:
                    continue
    except Exception:
        pass
    return ids

def _migrate_forums_to_jsonl() -> None:
    try:
        existing_ids = _load_conversation_ids()
        for path in FORUMS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            conv_id = data.get("id") or str(uuid.uuid4())
            if conv_id in existing_ids:
                continue
            question = (data.get("question") or "").strip()
            transcript = data.get("transcript") or []
            answers: List[Dict[str, Any]] = []
            if isinstance(transcript, list):
                for item in transcript:
                    if isinstance(item, dict):
                        answers.append({
                            "model": str(item.get("model") or ""),
                            "ts": str(item.get("ts") or _now_iso()),
                            "response": str(item.get("message") or ""),
                        })
            record: Dict[str, Any] = {
                "id": conv_id,
                "question": question,
                "answers": answers,
                "status": str(data.get("status") or "complete"),
            }
            _append_conversation_jsonl(record)
    except Exception:
        pass

# --- Data models ---
class PostStatus(str, Enum):
    thinking = "thinking"
    success = "success"
    fail = "fail"
    error = "error"
    
class ForumCreateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    models: List[str] = Field(..., min_items=1, description="List of model IDs to participate")

class ForumPost(BaseModel):
    model: str
    message: str
    status: PostStatus = Field(default=PostStatus.thinking)
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ForumCreateResponse(BaseModel):
    id: str

class ForumDetailResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    id: str
    question: str
    models: List[str]
    status: ForumStatus
    transcript: Dict[str, str]

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

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Helper functions ---
def _system_prompt(question: str, max_chars: int) -> str:
    template = _get_system_template()
    return template.safe_substitute(
        QUESTION=question,
        MAX_CHARACTERS=str(max_chars),
    )

def _load_template(template_name: str) -> Template:
    path = TEMPLATES_DIR / template_name
    text = path.read_text(encoding="utf-8")
    return Template(text)

def _truncate_question_to_input_budget(question: str) -> str:
    # Approximate tokens as chars/4 and limit accordingly
    char_budget = max(1, max_input * approx_chars_per_token)
    q = (question or "").strip()
    if len(q) <= char_budget:
        return q
    return q[:char_budget].rstrip()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _load_enabled_models() -> List[str]:
    """Load enabled model ids from models.yaml."""
    try:
        models_path = PROJECT_ROOT / "models.yaml"
        if not models_path.exists():
            return []
        cfg = yaml.safe_load(models_path.read_text(encoding="utf-8"))
        items = cfg.get("models", []) if isinstance(cfg, dict) else []
        enabled = [m.get("id") for m in items if m and m.get("enabled")]
        return [m for m in enabled if isinstance(m, str)]
    except Exception:
        return []

def _slugify_for_id(text: str) -> str:
    # Allow only alnum, dash, underscore; replace others with '-'
    return ''.join(ch if (ch.isalnum() or ch in ['-','_']) else '-' for ch in text)

async def _model_reply(
    model_name: str,
    question: str,
) -> ForumPost:
    system = _system_prompt(question=question, max_chars=max_chars)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": question})

    try:
        response = await asyncio.wait_for(
            acompletion(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
            ),
            timeout=max_time,
        )
    except asyncio.TimeoutError:
        return ForumPost(model=model_name, message="[timeout]", status=PostStatus.fail)
    except Exception as e:
        return ForumPost(
            model=model_name,
            message=f"[error] {type(e).__name__}: {str(e)}",
            status=PostStatus.error,
        )

    def _extract_content(resp: Any) -> str:
        """Return the assistant content using the OpenAI-style schema: choices[0].message.content"""
        try:
            if hasattr(resp, "model_dump") and callable(getattr(resp, "model_dump")):
                resp = resp.model_dump()
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""

    content = _extract_content(response)
    if not content:
        return ForumPost(model=model_name, message="[empty]", status=PostStatus.fail, ts=_now_iso())
    if len(content) > max_chars:
        truncated = content[:max_chars].rstrip()
        return ForumPost(model=model_name, message=truncated, status=PostStatus.fail, ts=_now_iso())
    return ForumPost(model=model_name, message=content, status=PostStatus.success, ts=_now_iso())

# In-memory store for forums (still used for in-flight generation)
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
def _to_detail_response(forum: Forum) -> ForumDetailResponse:
    unpacked: Dict[str, str] = {post.model: post.message for post in forum.transcript}
    return ForumDetailResponse(
        id=forum.id,
        question=forum.question,
        models=forum.models,
        status=forum.status,
        transcript=unpacked,
    )

def _get_forum_by_id(forum_id: str) -> Forum:
    forum = FORUMS.get(forum_id)
    if forum is None:
        loaded = _load_forum_from_disk(forum_id)
        if loaded is None:
            raise HTTPException(status_code=404, detail="Forum not found")
        forum = loaded
        # Store back in memory for quick future access
        async def _store():
            async with FORUMS_LOCK:
                FORUMS[forum_id] = loaded
        try:
            # Fire and forget; storing in-memory is best-effort here
            asyncio.create_task(_store())
        except Exception:
            pass
    return forum

def _create_forum_internal(question: str, selected_models: List[str]) -> Forum:
    bounded_question = _truncate_question_to_input_budget(question)
    transcript: List[ForumPost] = [
        ForumPost(model=m, message="[pending]", status=PostStatus.thinking, ts=_now_iso())
        for m in selected_models
    ]
    forum_id = str(uuid.uuid4())
    forum = Forum(
        id=forum_id,
        question=bounded_question,
        models=selected_models,
        transcript=transcript,
        status=ForumStatus.pending,
        created_at=_now_iso(),
    )
    return forum

def _render_models_checkboxes(models: List[str]) -> str:
    if not models:
        return '<div class="models"><em>No models available</em></div>'
    checkboxes = []
    for model_id in models:
        safe_model_id = html_escape(model_id)
        checkboxes.append(
            f'<label class="model-pill"><input type="checkbox" name="models" value="{safe_model_id}" checked> {safe_model_id}</label>'
        )
    return '<div class="models">' + "\n".join(checkboxes) + "</div>"

def _render_post(forum_id: str, post: ForumPost) -> str:
    template = _load_template("post.html")
    safe_model = html_escape(post.model)
    safe_msg = html_escape(post.message)
    safe_ts = html_escape(post.ts or _now_iso())
    status_class = html_escape(f"status-{post.status}")
    model_slug = _slugify_for_id(post.model)
    post_id = f"post-{forum_id}-{model_slug}"
    model_url = url_quote(post.model, safe='')
    avatar_initials = html_escape(((post.model.split("/")[-1] or "?")[:2].upper()))
    return template.safe_substitute(
        post_id=html_escape(post_id),
        status_class=status_class,
        avatar_initials=avatar_initials,
        safe_model=safe_model,
        safe_ts=safe_ts,
        safe_msg=safe_msg,
        forum_id=html_escape(forum_id),
        model_url=model_url,
    )

def _render_post_edit(forum_id: str, post: ForumPost) -> str:
    template = _load_template("post_edit.html")
    safe_msg = html_escape(post.message)
    model_slug = _slugify_for_id(post.model)
    post_id = f"post-{forum_id}-{model_slug}"
    model_url = url_quote(post.model, safe='')
    return template.safe_substitute(
        post_id=html_escape(post_id),
        forum_id=html_escape(forum_id),
        model_url=model_url,
        max_chars=str(max_chars),
        safe_msg=safe_msg,
    )

def _render_forum_card(forum: Forum) -> str:
    template = _load_template("forum_card.html")
    safe_q = html_escape(forum.question)
    created = html_escape(forum.created_at)
    status = forum.status
    status_marker = "<span class=\"status-complete\"></span>" if status in (ForumStatus.complete, ForumStatus.error) else ""
    posts_html = "\n".join(_render_post(forum.id, p) for p in forum.transcript)
    return template.safe_substitute(
        forum_id=html_escape(forum.id),
        created=created,
        safe_q=safe_q,
        posts_html=posts_html,
        status=html_escape(status),
        status_marker=status_marker,
    )

def _render_index_html() -> str:
    template = _load_template("index.html")
    enabled_models = _load_enabled_models()
    model_list_html = _render_models_checkboxes(enabled_models)
    return template.safe_substitute(models=model_list_html)
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
                transcript.append(
                    ForumPost(
                        model=model_name,
                        message=f"[error] {type(result).__name__}: {str(result)}",
                        status=PostStatus.error,
                    )
                )
            else:
                # result is already a ForumPost
                transcript.append(result)

        # Persist final conversation into JSONL
        record: Dict[str, Any] = {
            "id": forum_id,
            "question": question,
            "answers": [
                {
                    "model": p.model,
                    "ts": p.ts,
                    "response": p.message,
                }
                for p in transcript
            ],
            "status": str(ForumStatus.complete),
        }
        _append_conversation_jsonl(record)
        async with FORUMS_LOCK:
            forum = FORUMS.get(forum_id)
            if forum is not None:
                forum.transcript = transcript
                forum.status = ForumStatus.complete
    except Exception as e:
        async with FORUMS_LOCK:
            forum = FORUMS.get(forum_id)
            if forum is not None:
                forum.transcript = transcript
                forum.status = ForumStatus.error

@app.post("/forum", response_class=HTMLResponse)
async def create_forum(req: Request) -> HTMLResponse:
    form = await req.form()
    question = (form.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    raw_models = form.getlist("models") if hasattr(form, "getlist") else []
    selected_models = [m for m in raw_models if isinstance(m, str) and m.strip()]
    if not selected_models:
        selected_models = _load_enabled_models()
    bounded_question = _truncate_question_to_input_budget(question)
    forum = _create_forum_internal(question=bounded_question, selected_models=selected_models)
    async with FORUMS_LOCK:
        FORUMS[forum.id] = forum
    asyncio.create_task(
        _generate_forum_transcript(
            forum_id=forum.id,
            question=forum.question,
            model_names=forum.models,
        )
    )
    return HTMLResponse(content=_render_forum_card(forum))

@app.get("/forum", response_class=HTMLResponse)
async def get_forum_page() -> HTMLResponse:
    # Ensure store and perform one-time migration
    _ensure_conversations_file()
    _migrate_forums_to_jsonl()
    return HTMLResponse(content=_render_index_html())

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def root_redirect() -> HTMLResponse:
    return HTMLResponse(content=_render_index_html())