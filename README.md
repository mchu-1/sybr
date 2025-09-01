# Debate Arena
A multi-turn debate API between two frontier models.

## Features
1. Randomly assigns sides: affirmative vs negative
2. Two different LLMs are selected
3. Each model sees the full conversation history before responding
4. 280-character responses enforced via `max_tokens` approximation

## Requirements
- Python 3.10+
- Models and keys are managed by LiteLLM

## Run
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoint: POST /debate
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
- If two or more models are enabled, two distinct models are chosen at random.
- If exactly one model is enabled, it will debate itself.
- You may still pass `models` in the request body to override.

### Conversation
Hyperparameters live in `config.yaml`.

### Model configuration
Models are listed in `models.yaml`.