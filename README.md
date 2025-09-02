# Forum Arena
A multi-turn forum API where multiple models post in a shared thread.

## Features
1. Multiple models each post once in a round-robin forum (no adversarial sides)
2. Each model sees the full conversation history before responding
3. Fixed small output budget via `max_characters`
4. Fixed small input budget via `max_input_tokens` truncation
5. Expanded reasoning budget via `max_reasoning_tokens` (~10K tokens)
6. Threads are stored in-memory and retrievable by UUID for the session

## Requirements
- Python 3.10+
- Models and keys are managed by LiteLLM

## Run
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoint: POST /forums
Request body:
```json
{
  "question": "What is the meaning of life?",
  "models": [
    "gpt-5-high",
    "grok-4"
  ]
}
```

Response (ForumThread object):
```json
{
  "id": "c2c5d57b-5b3f-4bd9-a9f3-7f9f6b8b4a8e",
  "question": "What is the meaning of life?",
  "models": ["gpt-5-high", "grok-4"],
  "transcript": [
    {
      "turn": 1,
      "model": "gpt-5-high",
      "message": "...280 chars max..."
    },
    {
      "turn": 1,
      "model": "grok-4",
      "message": "...280 chars max..."
    }
  ]
}
```

### Model selection
- Provide one or more model IDs in an array.
- Model IDs must be enabled in `models.yaml`.
- You may include duplicates; duplicates are de-duplicated.

### Conversation
Hyperparameters live in `config.yaml`. Each model responds once per thread.

## Endpoint: GET /forums/{id}
Returns the stored ForumThread object for the given UUID. In-memory only for the life of the process.

### Model configuration
Models are listed in `models.yaml`.