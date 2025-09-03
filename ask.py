import asyncio
import argparse
import json
import uuid
import re
import os
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv, dotenv_values
from litellm import acompletion
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import PlainTextResponse, Response
from twilio.request_validator import RequestValidator
from twilio.rest import Client


# --- Setup ---
load_dotenv()
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
MODELS_PATH: Path = PROJECT_ROOT / "models.yaml"
SYSTEM_PROMPT_PATH: Path = PROJECT_ROOT / "system.md"
FORUM_JSONL_PATH: Path = PROJECT_ROOT / "forum.jsonl"
# Debate role prompts
DEBATE_PROMPT_PATH: Path = PROJECT_ROOT / "debate.md"
VERDICT_PROMPT_PATH: Path = PROJECT_ROOT / "verdict.md"

# --- FastAPI (consolidated server) ---
app = FastAPI()


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


# --- Environment / Twilio ---

ENV_MAP: Dict[str, str] = {k: v for k, v in (dotenv_values() or {}).items() if isinstance(k, str)}


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = ENV_MAP.get(key)
    if val is None or str(val).strip() == "":
        return os.getenv(key, default)
    return str(val)


TWILIO_ACCOUNT_SID: str = _env("TWILIO_ACCOUNT_SID", "") or ""
TWILIO_AUTH_TOKEN: str = _env("TWILIO_AUTH_TOKEN", "") or ""
TWILIO_FROM_NUMBER: str = _env("TWILIO_FROM_NUMBER", "") or ""
TWILIO_WHITELIST: str = _env("TWILIO_WHITELIST", "") or ""
FORUM_URL: str = _env("FORUM_URL", "https://thesybr.net") or "https://thesybr.net"
TWILIO_VALIDATE: bool = str(_env("TWILIO_VALIDATE", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}

_twilio_client_singleton: Optional[Client] = None


def _get_twilio_client() -> Client:
    global _twilio_client_singleton
    if _twilio_client_singleton is None:
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            raise RuntimeError("Twilio credentials are not configured")
        _twilio_client_singleton = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _twilio_client_singleton


def _parse_whitelist(raw: str) -> List[str]:
    if not raw:
        return []
    items = re.split(r"[,;\s]+", str(raw))
    return [x.strip() for x in items if x and x.strip()]


def _is_whitelisted_number(number: str) -> bool:
    allowed = _parse_whitelist(TWILIO_WHITELIST)
    if not allowed:
        return False
    if "*" in allowed:
        return True
    normalized = (number or "").strip()
    return normalized in allowed


def _validate_twilio_signature(request: Request, form_data: Dict[str, str]) -> bool:
    if not TWILIO_VALIDATE:
        return True
    signature = request.headers.get("X-Twilio-Signature") or ""
    if not signature or not TWILIO_AUTH_TOKEN:
        return False
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    # Full URL as Twilio sees it
    url = str(request.url)
    try:
        return bool(validator.validate(url, form_data, signature))
    except Exception:
        return False


def _extract_majority_summary(record: Dict[str, Any]) -> str:
    vote = record.get("vote", {}) or {}
    yes = int(vote.get("for", 0) or 0)
    no = int(vote.get("against", 0) or 0)
    if yes > no:
        majority = "YES"
    elif no > yes:
        majority = "NO"
    else:
        majority = "SPLIT"

    # Try to craft a brief justification from the first majority verdict
    brief = ""
    answers = record.get("answers") or []
    for ans in answers:
        verdict = (ans or {}).get("verdict") or {}
        v = str(verdict.get("vote", "")).lower()
        if (majority == "YES" and v == "for") or (majority == "NO" and v == "against"):
            raw = str(verdict.get("verdict") or verdict.get("justification") or "").strip()
            if raw:
                # Heuristic: second sentence or trimmed text up to 200 chars
                m = re.findall(r"[^.!?]+[.!?]", raw)
                if len(m) >= 2:
                    brief = m[1].strip()
                else:
                    brief = raw[:200].strip()
            break

    summary = f"Majority: {majority}. YES: {yes}, NO: {no}."
    if brief:
        return f"{summary} Reason: {brief}"
    return summary


async def _process_and_reply(question: str, user_number: str) -> None:
    try:
        _ensure_forum_file()
        record = await _debate_all_models(question)
        _append_forum_record(record)
        # Compose SMS body in compact format:
        # Yes (X) No (Y)\nthesybr.net
        vote = record.get("vote", {}) or {}
        yes = int(vote.get("for", 0) or 0)
        no = int(vote.get("against", 0) or 0)
        domain = re.sub(r"^https?://", "", (FORUM_URL or "").strip()).split("/")[0] or "thesybr.net"
        body = f"yes: {yes} no: {no}\n{domain}"
        # Send SMS
        client = _get_twilio_client()
        if TWILIO_FROM_NUMBER:
            client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=user_number)
    except Exception as e:
        try:
            client = _get_twilio_client()
            fallback = f"sybr — error answering your question: {type(e).__name__}. Please try again later."
            if TWILIO_FROM_NUMBER:
                client.messages.create(body=fallback, from_=TWILIO_FROM_NUMBER, to=user_number)
        except Exception:
            # Swallow secondary failures
            pass


@app.post("/sms")
async def sms_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    from_number: str = Form("", alias="From"),
    body: str = Form("", alias="Body"),
    to_number: str = Form("", alias="To"),
) -> Response:
    # Validate signature if enabled
    form = await request.form()
    form_map: Dict[str, str] = {k: str(v) for k, v in dict(form).items()}
    if not _validate_twilio_signature(request, form_map):
        return Response(content="", status_code=403)

    # Whitelist check
    if not _is_whitelisted_number(from_number):
        return Response(content="", status_code=204)

    question = (body or "").strip()
    if not question:
        return Response(content="", status_code=204)

    # Queue background processing and return no content (no auto-reply)
    background_tasks.add_task(_process_and_reply, question, from_number)
    return Response(content="", status_code=204)


# --- Config ---


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
    try:
        return {
            "max_tokens": int(cfg["max_tokens"]),
            "max_time": int(cfg["max_time"]),
        }
    except KeyError as e:
        raise RuntimeError(f"Missing required config key: {e.args[0]}")


def _load_enabled_models() -> List[str]:
    cfg = _read_yaml(MODELS_PATH)
    items = cfg.get("models", []) if isinstance(cfg, dict) else []
    enabled = [m.get("id") for m in items if isinstance(m, dict) and m.get("enabled")]
    return [m for m in enabled if isinstance(m, str) and m.strip()]


def _read_text(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def _system_prompt(question: str) -> str:
    # Backwards-compatible default system prompt (unused in debate flow)
    tmpl_text = _read_text(SYSTEM_PROMPT_PATH)
    tmpl = Template(tmpl_text)
    return tmpl.safe_substitute(QUESTION=question)


def _role_prompt(path: Path) -> str:
    return _read_text(path)


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


async def _ask_agent(
    model_name: str,
    system_text: str,
    user_text: str,
    max_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
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

    content = _extract_content(response)
    try:
        hidden = getattr(response, "_hidden_params", {}) or {}
        cost_val = float(hidden.get("response_cost", 0.0))
    except Exception:
        cost_val = 0.0
    if not content:
        return {"model": model_name, "ts": _now_iso(), "response": "[empty]", "cost": cost_val}
    return {"model": model_name, "ts": _now_iso(), "response": content, "cost": cost_val}


async def _debate_all_models(question: str) -> Dict[str, Any]:
    cfg = _load_config()
    models = _load_enabled_models()
    if not models:
        raise RuntimeError("No enabled models found in models.yaml")

    # 1) Arguments: each model argues FOR and AGAINST as independent agents, using consolidated debate prompt
    debate_prompt = _role_prompt(DEBATE_PROMPT_PATH)

    aff_tasks = [
        _ask_agent(
            model_name=m,
            system_text=debate_prompt,
            user_text=(
                f"Question:\n{question}\n\n"
                f"Side: Affirmative\n"
            ),
            max_tokens=cfg["max_tokens"],
            timeout_s=cfg["max_time"],
        )
        for m in models
    ]
    neg_tasks = [
        _ask_agent(
            model_name=m,
            system_text=debate_prompt,
            user_text=(
                f"Question:\n{question}\n\n"
                f"Side: Negative\n"
            ),
            max_tokens=cfg["max_tokens"],
            timeout_s=cfg["max_time"],
        )
        for m in models
    ]

    aff_results, neg_results = await asyncio.gather(
        asyncio.gather(*aff_tasks, return_exceptions=False),
        asyncio.gather(*neg_tasks, return_exceptions=False),
    )

    # Collate de-identified arguments (kept in-memory for adjudication)
    collated_for: List[str] = [r.get("response", "") for r in aff_results]
    collated_against: List[str] = [r.get("response", "") for r in neg_results]

    # 2) Adjudication: each model judges the collated arguments
    verdict_prompt = _role_prompt(VERDICT_PROMPT_PATH)
    aff_section = "\n".join([f"- {arg}" for arg in collated_for if arg])
    neg_section = "\n".join([f"- {arg}" for arg in collated_against if arg])
    verdict_user = (
        f"Question:\n{question}\n\n"
        f"Affirmative:\n{aff_section or '- [none]'}\n\n"
        f"Negative:\n{neg_section or '- [none]'}\n\n"
    )
    v_tasks = [
        _ask_agent(
            model_name=m,
            system_text=verdict_prompt,
            user_text=verdict_user,
            max_tokens=cfg["max_tokens"],
            timeout_s=cfg["max_time"],
        )
        for m in models
    ]
    verdict_results = await asyncio.gather(*v_tasks, return_exceptions=False)

    # Parse verdicts
    votes_for = 0
    votes_against = 0
    parsed_verdicts: List[Dict[str, Any]] = []
    for v in verdict_results:
        raw = (v.get("response") or "").strip()
        vote_val = "against"  # default to a concrete side to satisfy YES/NO requirement
        justification_val = raw
        # First, try to parse as the new soft sentence: "Yes. Reason." or "No. Reason."
        m_soft = re.match(r"^\s*(yes|no)[\s\t]*\.[\s\t]*(.+?)\.?\s*$", raw, flags=re.IGNORECASE | re.DOTALL)
        if m_soft:
            yn = m_soft.group(1).strip().lower()
            reason = m_soft.group(2).strip()
            vote_val = "for" if yn == "yes" else "against"
            justification_val = reason
        else:
            try:
                data = json.loads(raw)
                vote_raw = str(data.get("vote", "")).strip().lower()
                if vote_raw in {"yes", "no"}:
                    vote_val = "for" if vote_raw == "yes" else "against"
                elif vote_raw in {"for", "against"}:
                    vote_val = vote_raw
                justification_val = str(data.get("verdict", "") or data.get("justification", "")).strip() or justification_val
            except Exception:
                # Heuristic fallback: look for keywords
                txt = raw.lower()
                # Try to extract an inline vote field even if extra text surrounds JSON
                m = re.search(r'"vote"\s*:\s*"(yes|no|for|against)"', raw, flags=re.IGNORECASE)
                if m:
                    val = m.group(1).lower()
                    vote_val = "for" if val in {"yes", "for"} else "against"
                elif re.search(r'\byes\b|\baffirmative\b', txt):
                    vote_val = "for"
                elif re.search(r'\bno\b|\bnegative\b', txt):
                    vote_val = "against"
                elif re.search(r'\bfor\b', txt) and not re.search(r'\bagainst\b', txt):
                    vote_val = "for"
                elif re.search(r'\bagainst\b', txt) and not re.search(r'\bfor\b', txt):
                    vote_val = "against"

        if vote_val == "for":
            votes_for += 1
        elif vote_val == "against":
            votes_against += 1

        parsed_verdicts.append({
            "model": v.get("model"),
            "ts": v.get("ts"),
            "verdict": {"vote": vote_val, "verdict": justification_val},
            "cost": v.get("cost", 0.0),
        })

    # Merge per-model answers
    answers: List[Dict[str, Any]] = []
    for idx, m in enumerate(models):
        a = aff_results[idx]
        n = neg_results[idx]
        # Find corresponding verdict by model
        v = next((p for p in parsed_verdicts if p.get("model") == m), None)
        if v is None or not v.get("verdict"):
            raise RuntimeError(f"Missing adjudicator verdict for model: {m}")
        verdict_obj = v.get("verdict")
        # Basic validation: ensure required keys are present and vote is valid
        vote_val = str(verdict_obj.get("vote", "")).lower()
        reason_val = str(verdict_obj.get("verdict", "")).strip()
        if vote_val not in {"for", "against"} or not reason_val:
            raise RuntimeError(f"Invalid adjudicator verdict for model {m}: {verdict_obj}")

        answers.append({
            "model": m,
            "for": {
                "content": a.get("response", ""),
                "ts": a.get("ts"),
            },
            "against": {
                "content": n.get("response", ""),
                "ts": n.get("ts"),
            },
            "verdict": verdict_obj,
        })

    # Build record
    record: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answers": answers,
        "vote": {"for": votes_for, "against": votes_against},
        "status": "complete",
        "created_at": _now_iso(),
    }

    # Sum total conversation cost
    try:
        record["cost"] = float(
            sum(float(x.get("cost", 0.0)) for x in aff_results)
            + sum(float(x.get("cost", 0.0)) for x in neg_results)
            + sum(float(x.get("cost", 0.0)) for x in verdict_results)
        )
    except Exception:
        record["cost"] = 0.0

    return record


def _print_record_to_cli(record: Dict[str, Any]) -> None:
    vote_obj = record.get("vote", {}) or {}
    affirmative_count = int(vote_obj.get("for", 0) or 0)
    negative_count = int(vote_obj.get("against", 0) or 0)
    print(f"Affirmative: {affirmative_count}")
    print(f"Negative: {negative_count}")


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
    record = asyncio.run(_debate_all_models(question))
    _append_forum_record(record)
    _print_record_to_cli(record)


if __name__ == "__main__":
    main()


