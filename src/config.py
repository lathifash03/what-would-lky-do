"""Configuration. Models come from .env and are never hardcoded in src/."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class MissingConfig(RuntimeError):
    pass


def require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise MissingConfig(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def chat_model() -> str:
    return require("OPENAI_MODEL")


def embed_model() -> str:
    return require("EMBED_MODEL")


def prompt_text(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


def prompt_hash(name: str) -> str:
    """Short hash of the prompt file, recorded in every run log.

    A run log that does not say which prompt produced it cannot be reproduced.
    """
    import hashlib

    return hashlib.sha256(prompt_text(name).encode("utf-8")).hexdigest()[:12]
