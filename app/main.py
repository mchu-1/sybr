import random

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
class DebateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    models: Optional[List[str]] = None
    turns: int = Field(
        default_turns,
        ge=1,
        description="Number of turns per side",
    )

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

# --- Helper functions ---
def _select_two_models(_: Optional[List[str]]) -> List[str]:
    # Always select from models.yaml. If exactly one model, it debates itself.
    yaml_models = _load_models_from_yaml()
    if len(yaml_models) >= 2:
        return random.sample(yaml_models, 2)
    if len(yaml_models) == 1:
        return [yaml_models[0], yaml_models[0]]

    raise HTTPException(status_code=400, detail=(
        "No models available. Enable at least one model in models.yaml."
    ))

def _assign_sides() -> Dict[str, Side]:
    sides: List[Side] = ["affirmative", "negative"]
    random.shuffle(sides)
    return {"speaker_a": sides[0], "speaker_b": sides[1]}

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

# --- Endpoints ---
@app.post("/debate", response_model=List[DebateTurn])
async def debate(req: DebateRequest) -> List[DebateTurn]:
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    model_a, model_b = _select_two_models(req.models)
    sides_map = _assign_sides()  # speaker_a / speaker_b → side

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

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}