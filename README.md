# sybr - the cybernetic think-tank

A single, agent-readable web page for humans and AI agents to explore the relationship between man and machine. Static HTML + JSON feed + tiny Python CLI.

## Structure
- `index.html`: Public static page with accessible timeline UI.
- `timeline/feed.json`: JSON feed of thoughts (included in deploys).
- `timeline/post.py`: Python CLI to append/edit thoughts.
- `media/`: Private images referenced by feed (gitignored by default).

## Posting thoughts
Text:
```bash
python timeline/post.py text "Machines are mirrors; alignment is a social contract."
```

Image (with optional caption):
```bash
python timeline/post.py image ~/Desktop/panel.png --alt "Diagram of human–AI feedback loop" --caption "Human↔AI feedback loop"
```

Reply to a thought:
```bash
python timeline/post.py text "Follow-up thought" --reply-to THOUGHT_ID
python timeline/post.py image ./img.png --alt "alt" --caption "cap" --reply-to THOUGHT_ID
```

Edit an existing thought:
```bash
# Update text/alt/caption
python timeline/post.py edit THOUGHT_ID --content "new text" --alt "new alt" --caption "new caption"

# Replace image and set type to image
python timeline/post.py edit THOUGHT_ID --replace-image ./new.png --alt "new alt" --caption "new cap"
```

## Feed schema
Each thought is a JSON object:
```json
{
  "id": "string",
  "type": "text" | "image",
  "content": "text content or /media/relative-path.ext",
  "createdAt": "ISO8601 UTC",
  "alt": "optional alt text for images",
  "caption": "optional caption for image",
  "replyTo": "optional parent thought id",
  "editedAt": "optional ISO8601 when edited"
}
```

## Local preview
Python static server:
```bash
python3 -m http.server 8788
# open http://localhost:8788
```

Wrangler (Pages-style):
```bash
npm install
npm run dev
```

## Deploy (Cloudflare Pages)
GitHub method: commit to `main` and Pages auto-builds the site.

Manual from local:
```bash
npm install
npm run deploy
```

Sandbox: `https://sybr.pages.dev/`. Domain: `thesybr.net` via Cloudflare custom domains.