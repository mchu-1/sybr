# sybr - the cybernetic think-tank

A single, agent-readable web page for humans and AI agents to explore the relationship between man and machine. Static HTML + private JSON feed.

## Structure

- `index.html`: Public static page with accessible timeline UI.
- `timeline/feed.json`: Private JSON feed of thoughts (gitignored).
- `timeline/post.py`: Python CLI to append thoughts.
- `media/`: Private images referenced by feed (gitignored).

## Privacy / tracking

This repo open-sources the template but keeps contents private via `.gitignore`:

- `timeline/feed.json`
- `media/`
- `.env`
- `.wrangler/`

## Posting thoughts (developer-only)

Text:

```bash
python timeline/post.py text "Machines are mirrors; alignment is a social contract."
```

Image:

```bash
python timeline/post.py image ~/Desktop/panel.png --alt "Diagram of human-AI feedback loop"
```

This copies the image into `media/` and appends a new entry to `timeline/feed.json` with an automatic UTC timestamp.

## Local preview

Using Cloudflare Wrangler (Node 18+):

```bash
npm install
npm run dev
```

Then open the local URL printed by Wrangler. The public site will show "No thoughts yet" if your private feed is not present.

## Deploy (Cloudflare Pages)

Authenticate and create project once:

```bash
npm run cf:login
npm run cf:project:create
```

Deploy current directory as a static site:

```bash
npm run deploy
```

Notes:
- The deployed bundle is the current working directory. The repo ignores `timeline/feed.json` and `media/`, but they will be deployed if present on disk when you run deploy.
- Domain: `thesybr.net` — map this Pages project to the domain in Cloudflare dashboard.

## Feed schema

Each thought is a JSON object:

```json
{
  "id": "string",
  "type": "text" | "image",
  "content": "text content or /media/relative-path.ext",
  "createdAt": "ISO8601 UTC",
  "alt": "optional alt text for images"
}
```

## Roadmap (not implemented)

- Social integrations (Mastodon, Bluesky, X) — reserved for later.
- Multi-author support — currently single author by design.
- Feeds (RSS/Atom/JSON Feed) for public excerpts only.
