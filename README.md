# Forum Arena
An API where multiple models each provide a single, independent answer to a question.

## Features
1. Multiple models each post once, independently (no shared history)
2. Each model does not see other answers; responses are independent
3. Fixed small output budget via `max_characters`
4. Fixed small input budget via `max_input_tokens` truncation
5. Expanded reasoning budget via `max_reasoning_tokens` (~10K tokens)
6. Forums are stored in-memory and retrievable by UUID for the session

## Requirements
- Python 3.10+
- Models and keys are managed by LiteLLM

## Run
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Auth & Security
- POST `/forum` requires a Bearer token. Set an environment variable `FORUM_AUTH_TOKEN` on the server (e.g., in a local `.env`).
- CORS is restricted to `https://sybr.pages.dev` and common local dev origins.

Example `.env`:
```bash
FORUM_AUTH_TOKEN=your-strong-random-token
```

Example request with token:
```bash
curl -X POST "http://localhost:8000/forum" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-strong-random-token" \
  -d '{
    "question": "What is the meaning of life?",
    "models": ["gpt-5", "grok-4"]
  }'
```

## Endpoint: POST /forum
Request body:
```json
{
  "question": "What is the meaning of life?",
  "models": [
    "gpt-5",
    "grok-4"
  ]
}
```

Response (Forum object):
```json
{
  "id": "c2c5d57b-5b3f-4bd9-a9f3-7f9f6b8b4a8e",
  "question": "What is the meaning of life?",
  "models": ["gpt-5", "grok-4"],
  "transcript": [
    {
      "model": "gpt-5",
      "message": "...280 chars max..."
    },
    {
      "model": "grok-4",
      "message": "...280 chars max..."
    }
  ]
}
```

### Model selection
- Provide one or more model IDs in an array.
- Model IDs must be enabled in `models.yaml`.
- At least 1 model must be selected, up to the number of available models.
- You may include duplicates; duplicates are de-duplicated.

### Conversation
Hyperparameters live in `config.yaml`. Each model responds once per forum, independently and without shared context.

## Endpoint: GET /forum/{forum_id}
Returns the stored Forum object for the given UUID. In-memory only for the life of the process.

### Model configuration
Models are listed in `models.yaml`.