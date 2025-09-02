import asyncio
import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from litellm import acompletion


# --- Setup ---
load_dotenv()
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
MODELS_PATH: Path = PROJECT_ROOT / "models.yaml"
SYSTEM_PROMPT_PATH: Path = PROJECT_ROOT / "system.md"
FORUM_JSONL_PATH: Path = PROJECT_ROOT / "forum.jsonl"


# --- Config ---
approx_chars_per_token: int = 4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Required file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML at {path} must be a mapping at the top-level")
    return data


def _load_config() -> Dict[str, Any]:
    cfg = _read_yaml(CONFIG_PATH)
    return {
        "max_tokens": int(cfg.get("max_tokens", 1000)),
        "max_input": int(cfg.get("max_input", 64)),
        "max_time": int(cfg.get("max_time", 30)),
    }


def _load_enabled_models() -> List[str]:
    cfg = _read_yaml(MODELS_PATH)
    items = cfg.get("models", []) if isinstance(cfg, dict) else []
    enabled = [m.get("id") for m in items if isinstance(m, dict) and m.get("enabled")]
    return [m for m in enabled if isinstance(m, str) and m.strip()]


def _truncate_question(question: str, max_input_tokens: int) -> str:
    budget = max(1, max_input_tokens * approx_chars_per_token)
    q = (question or "").strip()
    if len(q) <= budget:
        return q
    return q[:budget].rstrip()


def _system_prompt(question: str) -> str:
    tmpl_text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    tmpl = Template(tmpl_text)
    return tmpl.safe_substitute(
        QUESTION=question,
    )


def _ensure_forum_file() -> None:
    if not FORUM_JSONL_PATH.exists():
        FORUM_JSONL_PATH.touch()


def _append_forum_record(record: Dict[str, Any]) -> None:
    with FORUM_JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _ask_one_model(model_name: str, question: str, max_tokens: int, timeout_s: int) -> Dict[str, Any]:
    system = _system_prompt(question=question)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    try:
        response = await asyncio.wait_for(
            acompletion(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        return {"model": model_name, "ts": _now_iso(), "response": "[timeout]", "cost": 0.0}
    except Exception as e:
        return {"model": model_name, "ts": _now_iso(), "response": f"[error] {type(e).__name__}: {str(e)}", "cost": 0.0}

    def _extract_content(resp: Any) -> str:
        try:
            if hasattr(resp, "model_dump") and callable(getattr(resp, "model_dump")):
                resp = resp.model_dump()
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""

    # Extract content and response cost (if provided by client)
    content = _extract_content(response)
    try:
        hidden = getattr(response, "_hidden_params", {}) or {}
        cost_val = float(hidden.get("response_cost", 0.0))
    except Exception:
        cost_val = 0.0
    if not content:
        return {"model": model_name, "ts": _now_iso(), "response": "[empty]", "cost": cost_val}
    return {"model": model_name, "ts": _now_iso(), "response": content, "cost": cost_val}


async def _ask_all_models(question: str) -> Dict[str, Any]:
    cfg = _load_config()
    models = _load_enabled_models()
    if not models:
        raise RuntimeError("No enabled models found in models.yaml")

    bounded_question = _truncate_question(question, cfg["max_input"]) 

    tasks = [
        _ask_one_model(
            model_name=m,
            question=bounded_question,
            max_tokens=cfg["max_tokens"],
            timeout_s=cfg["max_time"],
        )
        for m in models
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    record: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "question": bounded_question,
        "answers": results,
        "status": "complete",
        "created_at": _now_iso(),
    }
    # Sum total conversation cost across model responses
    try:
        record["cost"] = float(sum((float(a.get("cost", 0.0)) for a in results)))
    except Exception:
        record["cost"] = 0.0
    return record


def _print_record_to_cli(record: Dict[str, Any]) -> None:
    print(f"id: {record['id']}")
    print(f"question: {record['question']}")
    print("answers:")
    for ans in record.get("answers", []):
        model = ans.get("model", "?")
        response = ans.get("response", "")
        print(f"- {model}: {response}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask all models a question and append results to forum.jsonl")
    parser.add_argument("question", nargs="?", help="The question to ask")
    args = parser.parse_args()

    question = (args.question or "").strip()
    if not question:
        try:
            question = input("Question: ").strip()
        except EOFError:
            question = ""

    if not question:
        raise SystemExit("A question is required")

    _ensure_forum_file()
    record = asyncio.run(_ask_all_models(question))
    _append_forum_record(record)
    _print_record_to_cli(record)


if __name__ == "__main__":
    main()


