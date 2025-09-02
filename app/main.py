import uuid

from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
from string import Template

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from litellm import acompletion
import yaml
from dotenv import load_dotenv

Side = Literal["affirmative", "negative"]

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
class ModelAssignments(BaseModel):
    affirmative: str = Field(..., min_length=1)
    negative: str = Field(..., min_length=1)

class DebateCreateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    models: ModelAssignments
    turns: int = Field(
        default_turns,
        ge=1,
        description="Number of turns per side",
    )

class DebateTurn(BaseModel):
    turn: int
    side: Side
    model: str
    message: str

class Debate(BaseModel):
    id: str
    question: str
    affirmative: str
    negative: str
    transcript: List[DebateTurn]

app = FastAPI(title="Debate API", version="0.2.0")

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

def _system_prompt(side: Side, question: str) -> str:
    template = _get_system_template()
    return template.safe_substitute(
        SIDE=side.upper(),
        QUESTION=question,
        MAX_CHARACTERS=str(max_characters),
    )

async def _model_reply(
    model_name: str,
    side: Side,
    question: str,
    history_messages: List[Dict[str, str]],
) -> str:
    system = _system_prompt(side=side, question=question)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": question})
    messages.extend(history_messages)

    response = await acompletion(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
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

# In-memory store for debates
DEBATES: Dict[str, Debate] = {}

# --- Endpoints ---
@app.post("/debates", response_model=Debate)
async def create_debate(req: DebateCreateRequest) -> Debate:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    # Validate requested models exist and are enabled
    _validate_model_enabled(req.models.affirmative)
    _validate_model_enabled(req.models.negative)

    # Conversation history shared across both models
    history_messages: List[Dict[str, str]] = []
    transcript: List[DebateTurn] = []

    # Alternate sides deterministically: affirmative then negative per turn
    for turn_index in range(1, req.turns + 1):
        # Affirmative speaks first
        reply_affirmative = await _model_reply(
            model_name=req.models.affirmative,
            side="affirmative",
            question=req.question.strip(),
            history_messages=history_messages,
        )
        history_messages.append({"role": "assistant", "content": reply_affirmative})
        transcript.append(
            DebateTurn(
                turn=turn_index,
                side="affirmative",
                model=req.models.affirmative,
                message=reply_affirmative,
            )
        )

        # Negative responds, seeing the history including affirmative's reply
        reply_negative = await _model_reply(
            model_name=req.models.negative,
            side="negative",
            question=req.question.strip(),
            history_messages=history_messages,
        )
        history_messages.append({"role": "assistant", "content": reply_negative})
        transcript.append(
            DebateTurn(
                turn=turn_index,
                side="negative",
                model=req.models.negative,
                message=reply_negative,
            )
        )

    debate_id = str(uuid.uuid4())
    debate = Debate(
        id=debate_id,
        question=req.question.strip(),
        affirmative=req.models.affirmative,
        negative=req.models.negative,
        transcript=transcript,
    )

    DEBATES[debate_id] = debate
    return debate

@app.get("/debates/{debate_id}", response_model=Debate)
async def get_debate(debate_id: str) -> Debate:
    debate = DEBATES.get(debate_id)
    if debate is None:
        raise HTTPException(status_code=404, detail="Debate not found")
    return debate

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}