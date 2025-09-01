# sybr — local developer instructions

This file is local-only and should not be committed. It explains how to use and deploy the sybr page from your machine.

## 1) Prereqs
- Python 3.9+
- Node 18+
- Cloudflare account with Pages enabled
- GitHub repo `mchu-1/sybr` connected (optional if deploying via direct upload)

## 2) Post thoughts (local/private)
- Text:
```bash
python timeline/post.py text "A thought between human and machine."
```
- Image (copied into `media/`):
```bash
python timeline/post.py image /path/to/img.jpg --alt "Alt text for accessibility" --caption "optional caption"
```
- Reply to a thought:
```bash
python timeline/post.py text "Follow-up" --reply-to THOUGHT_ID
python timeline/post.py image ./img.png --alt "alt" --caption "cap" --reply-to THOUGHT_ID
```
- Edit a thought:
```bash
python timeline/post.py edit THOUGHT_ID --content "new text" --alt "new alt" --caption "new caption"
python timeline/post.py edit THOUGHT_ID --replace-image ./new.png --alt "new alt" --caption "new cap"
```
- The CLI appends to or updates `timeline/feed.json` with automatic UTC timestamps.

## 3) Preview locally
- Simple static server:
```bash
python3 -m http.server 8788
# open http://localhost:8788
```
- Cloudflare Pages-style preview:
```bash
npm install
npm run dev
```

## 4) Deploy to Cloudflare Pages
- GitHub (auto deploy on main):
```bash
git add -A
git commit -m "Update site"
git push
```
- Or manual deploy from local:
```bash
npm install
npm run deploy
```
- Sandbox: https://sybr.pages.dev/

## 5) Public content notes
- `timeline/feed.json` is included in deploys. Anything in it becomes public.
- `media/` is gitignored and excluded from deploys. If your feed references images (e.g., `/media/abc.jpg`), those will 404 unless you:
  - temporarily include specific images in the deploy, or
  - host images elsewhere and use full URLs in the feed, or
  - unignore `media/` (makes images public).

## 6) Domain
- Map `thesybr.net` to this Pages project in the Cloudflare dashboard.

## 7) Troubleshooting
- "No thoughts yet" → make sure `/timeline/feed.json` exists in the deployed build.
- Pages build settings: Framework None, Build command empty, Output directory `.`
- If deploy fails, re-auth with `npm run cf:login` and retry `npm run deploy`.
