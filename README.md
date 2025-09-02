# Debate Arena
A multi-turn debate API between two models with deterministic side assignments.

## Features
1. Deterministic sides: user selects models for affirmative and negative
2. Each model sees the full conversation history before responding
3. 280-character responses enforced via `max_tokens` approximation
4. Debates are stored in-memory and retrievable by UUID for the session

## Requirements
- Python 3.10+
- Models and keys are managed by LiteLLM

## Run
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoint: POST /debates
Request body:
```json
{
  "question": "Should governments ban TikTok?",
  "models": {
    "affirmative": "gpt-5-high",
    "negative": "grok-4"
  },
  "turns": 2
}
```

Response (Debate object):
```json
{
  "id": "c2c5d57b-5b3f-4bd9-a9f3-7f9f6b8b4a8e",
  "question": "Should governments ban TikTok?",
  "affirmative": "gpt-5-high",
  "negative": "grok-4",
  "transcript": [
    {
      "turn": 1,
      "side": "affirmative",
      "model": "gpt-5-high",
      "message": "...280 chars max..."
    },
    {
      "turn": 1,
      "side": "negative",
      "model": "grok-4",
      "message": "...280 chars max..."
    }
  ]
}
```

### Model selection
- You must provide model IDs for both `affirmative` and `negative`.
- Model IDs must be enabled in `models.yaml`.
- You may use the same model for both sides.

### Conversation
Hyperparameters live in `config.yaml`.

## Endpoint: GET /debates/{id}
Returns the stored Debate object for the given UUID. In-memory only for the life of the process.

### Model configuration
Models are listed in `models.yaml`.