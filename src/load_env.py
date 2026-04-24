import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def openai_key():
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def openai_model():
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()


def chroma_path():
    return str(_ROOT / ".chroma_data")
