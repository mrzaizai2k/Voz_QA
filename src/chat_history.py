from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path("data/history_chat.json")


def load_history() -> list:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_PATH.exists():
        try:
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            # Filter out empty chats on load
            return [c for c in data if c.get("messages")]
        except Exception:
            return []
    return []


def save_history(history: list):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Never save chats with no messages
    filtered = [c for c in history if c.get("messages")]
    HISTORY_PATH.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_chat(provider: str, model: str, max_pages: int, max_posts: int) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New chat",
        "created": datetime.now().isoformat(timespec="minutes"),
        "messages": [],
        "voz_url": "",
        "config": {
            "provider": provider,
            "model": model,
            "max_pages": max_pages,
            "max_posts": max_posts,
        },
    }


def get_chat_by_id(all_chats: list, chat_id: str) -> dict | None:
    for c in all_chats:
        if c["id"] == chat_id:
            return c
    return None


def upsert_chat(all_chats: list, chat: dict) -> list:
    """Replace existing chat or prepend if new."""
    for i, c in enumerate(all_chats):
        if c["id"] == chat["id"]:
            all_chats[i] = chat
            return all_chats
    all_chats.insert(0, chat)
    return all_chats