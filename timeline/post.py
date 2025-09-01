#!/usr/bin/env python3
"""
sybr timeline posting CLI

Usage:
  - Text thought:
      python timeline/post.py text "A human/AI co-thought"

  - Image thought (copied into media/):
      python timeline/post.py image /path/to/image.jpg --alt "Alt text"

This writes to timeline/feed.json with entries shaped as:
  {
    "id": "snowflake",
    "type": "text" | "image",
    "content": "string (text or relative /media/... path)",
    "createdAt": "ISO8601",
    "alt": "optional alt for images",
    "caption": "optional image caption",
    "replyTo": "optional parent thought id",
    "editedAt": "optional ISO8601 when edited"
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import shutil
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMELINE_DIR = PROJECT_ROOT / "timeline"
MEDIA_DIR = PROJECT_ROOT / "media"
FEED_PATH = TIMELINE_DIR / "feed.json"


def ensure_directories_exist() -> None:
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def load_feed() -> List[Dict[str, Any]]:
    if not FEED_PATH.exists():
        return []
    try:
        with FEED_PATH.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def write_feed(entries: List[Dict[str, Any]]) -> None:
    # Write atomically to avoid partial writes
    temp_path = FEED_PATH.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(FEED_PATH)


def generate_id() -> str:
    # UUID-based short id
    return uuid.uuid4().hex[:12]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compute_file_hash(path: Path) -> str:
    sha1 = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def copy_image_to_media(src: Path) -> Path:
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Image not found: {src}")
    file_hash = compute_file_hash(src)[:12]
    ext = src.suffix.lower() or ".bin"
    dest_name = f"{file_hash}{ext}"
    dest_path = MEDIA_DIR / dest_name
    if not dest_path.exists():
        shutil.copy2(src, dest_path)
    return dest_path


@dataclass
class Thought:
    id: str
    type: str  # 'text' | 'image'
    content: str
    createdAt: str
    alt: str | None = None
    caption: str | None = None
    replyTo: str | None = None
    editedAt: str | None = None

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("alt") is None:
            data.pop("alt", None)
        if data.get("caption") is None:
            data.pop("caption", None)
        if data.get("replyTo") is None:
            data.pop("replyTo", None)
        if data.get("editedAt") is None:
            data.pop("editedAt", None)
        return data


def add_text_thought(message: str) -> Thought:
    return Thought(
        id=generate_id(),
        type="text",
        content=message.strip(),
        createdAt=utc_now_iso(),
    )


def add_image_thought(image_path: Path, alt: str | None, caption: str | None) -> Thought:
    dest = copy_image_to_media(image_path)
    # Use site-root relative path for the client
    rel = "/" + str(dest.relative_to(PROJECT_ROOT)).replace(os.sep, "/")
    return Thought(
        id=generate_id(),
        type="image",
        content=rel,
        createdAt=utc_now_iso(),
        alt=(alt.strip() if alt else None),
        caption=(caption.strip() if caption else None),
    )


def main(argv: List[str]) -> int:
    ensure_directories_exist()

    parser = argparse.ArgumentParser(
        description="Append thoughts to sybr timeline feed (text or image)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_text = sub.add_parser("text", help="Post a text thought")
    p_text.add_argument("message", type=str, help="Text content of the thought")
    p_text.add_argument("--reply-to", dest="reply_to", type=str, default=None, help="Parent thought id to reply to")
    p_text.add_argument("--dry-run", action="store_true", help="Do not write to feed.json")

    p_image = sub.add_parser("image", help="Post an image thought")
    p_image.add_argument("path", type=str, help="Path to an image file to copy into media/")
    p_image.add_argument("--alt", type=str, default=None, help="Accessible alt text for the image")
    p_image.add_argument("--caption", type=str, default=None, help="Caption text to display with the image")
    p_image.add_argument("--reply-to", dest="reply_to", type=str, default=None, help="Parent thought id to reply to")
    p_image.add_argument("--dry-run", action="store_true", help="Do not write to feed.json")

    p_edit = sub.add_parser("edit", help="Edit an existing thought by id")
    p_edit.add_argument("id", type=str, help="ID of the thought to edit")
    p_edit.add_argument("--content", type=str, default=None, help="New content text (or replace image URL)")
    p_edit.add_argument("--alt", type=str, default=None, help="New alt text")
    p_edit.add_argument("--caption", type=str, default=None, help="New caption text")
    p_edit.add_argument("--replace-image", dest="replace_image", type=str, default=None, help="Path to new image file to copy into media/ and set as content")
    p_edit.add_argument("--dry-run", action="store_true", help="Do not write to feed.json")

    args = parser.parse_args(argv)

    feed = load_feed()

    if args.command == "text":
        thought = add_text_thought(args.message)
        if getattr(args, "reply_to", None):
            thought.replyTo = args.reply_to
    elif args.command == "image":
        thought = add_image_thought(Path(args.path), args.alt, args.caption)
        if getattr(args, "reply_to", None):
            thought.replyTo = args.reply_to
    elif args.command == "edit":
        target_id = args.id
        modified = False
        for item in feed:
            if item.get("id") == target_id:
                if args.replace_image:
                    new_path = copy_image_to_media(Path(args.replace_image))
                    item["content"] = "/" + str(new_path.relative_to(PROJECT_ROOT)).replace(os.sep, "/")
                    item["type"] = "image"
                    modified = True
                if args.content is not None:
                    item["content"] = args.content
                    modified = True
                if args.alt is not None:
                    if args.alt == "":
                        item.pop("alt", None)
                    else:
                        item["alt"] = args.alt
                    modified = True
                if args.caption is not None:
                    if args.caption == "":
                        item.pop("caption", None)
                    else:
                        item["caption"] = args.caption
                    modified = True
                if modified:
                    item["editedAt"] = utc_now_iso()
                break
        else:
            print(f"Thought not found: {target_id}", file=sys.stderr)
            return 1
        if getattr(args, "dry_run", False):
            print(json.dumps(feed, ensure_ascii=False, indent=2))
            return 0
        write_feed(feed)
        print(f"Edited thought {target_id} at {datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}")
        return 0
    else:
        parser.error("Unknown command")
        return 2

    new_feed = feed + [thought.to_json()]

    if getattr(args, "dry_run", False):
        print(json.dumps(new_feed, ensure_ascii=False, indent=2))
        return 0

    write_feed(new_feed)
    print(f"Appended {thought.type} thought {thought.id} at {thought.createdAt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


