### Debate API (FastAPI + LiteLLM + Tavily)

A simple multi-turn debate API between two different models using LiteLLM for model unification and Tavily for optional web research.

### Features
- Randomly assigns sides: affirmative vs negative
- Two different models selected from OpenAI, Anthropic, Google (Gemini), Grok
- Each model sees the full conversation history before responding
- 280-character responses enforced via `max_tokens` approximation
- Optional Tavily web search context

### Requirements
- Python 3.10+
- API keys (set only those you need):
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GEMINI_API_KEY` (or `GOOGLE_API_KEY` depending on your LiteLLM setup)
  - `XAI_API_KEY` (for Grok)
  - `TAVILY_API_KEY` (optional, for research)
- Optional environment variable to constrain model pool:
  - `DEBATE_MODELS`: comma-separated list, e.g. `gpt-4o-mini,claude-3-haiku-20240307`

### Install
```bash
python -m pip install -r requirements.txt
```

### Run
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Health Check
```bash
curl -s http://localhost:8000/healthz
```

### Endpoint: POST /debate
Request body:
```json
{
  "question": "Should governments ban TikTok?",
  "models": ["gpt-4o-mini", "claude-3-haiku-20240307"],
  "turns": 2,
  "use_search": true
}
```

Response (array of turns):
```json
[
  {
    "question": "Should governments ban TikTok?",
    "turn": 1,
    "side": "affirmative",
    "model": "gpt-4o-mini",
    "message": "...280 chars max..."
  },
  {
    "question": "Should governments ban TikTok?",
    "turn": 1,
    "side": "negative",
    "model": "claude-3-haiku-20240307",
    "message": "...280 chars max..."
  }
]
```

### Notes
- If you omit `models`, two are chosen from a default pool.
- If you set `DEBATE_MODELS`, they will be sampled from that list.
- `turns` is per model (e.g., 2 means A then B twice, total 4 replies).