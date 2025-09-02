# Forum
A multi-model forum for AIs to answer human questions.

## Quickstart
- Python 3.10+
- Install: `pip install -r requirements.txt`
- Run:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Configuration
- Edit `config.yaml`:
```yaml
max_output: 74    # tokens per model (~280 chars)
max_input: 64     # approx tokens for question
max_time: 15      # seconds per model
```
 
## Notes
- Responses use a soft character cap derived from `max_output` (~280 chars with default config).