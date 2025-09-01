# Debate Arena

A multi-turn debate API between two frontier models.

### Features
- Randomly assigns sides: affirmative vs negative
- Two different LLMs are selected
- Each model sees the full conversation history before responding
- 280-character responses enforced via `max_tokens` approximation

### Requirements
- Python 3.10+
- Models and keys are managed by LiteLLM via environment; see its docs.

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
curl -s http://localhost:8000/health
```

### Endpoint: POST /debate
Request body:
```json
{
  "question": "Should governments ban TikTok?",
  "models": ["gpt-5-high", "grok-4"],
  "turns": 2
}
```

Response (array of turns):
```json
[
  {
    "question": "Should governments ban TikTok?",
    "turn": 1,
    "side": "affirmative",
    "model": "gpt-5-high",
    "message": "...280 chars max..."
  },
  {
    "question": "Should governments ban TikTok?",
    "turn": 1,
    "side": "negative",
    "model": "grok-4",
    "message": "...280 chars max..."
  }
]
```

### Model selection
The app selects models from `models.yaml` by default.
- If two or more models are enabled, two distinct models are chosen at random.
- If exactly one model is enabled, it will debate itself.
- You may still pass `models` in the request body to override.
### Configuration (config.yaml)
Hyperparameters live in `config.yaml` at the repo root.


### Model config (models.yaml)
Models are listed in `models.yaml` at the repository root.

Notes: Only API keys are read from `.env`. Model IDs come from `models.yaml` (default) or the request body.