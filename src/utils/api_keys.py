"""Single source of truth for backend LLM API keys.

Keys live in a `.env` file at the repo root (gitignored) and are read via
environment variables — never from on-disk `*_API_KEY.txt` files. Importing this
module loads the `.env` once; callers use `get_api_key` / `has_api_key`.
"""

import os

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))  # idempotent; safe to import many times

_PROVIDER_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def get_api_key(provider: str) -> str:
    """Return the API key for `provider`, raising if it is unset."""
    var = _PROVIDER_ENV.get(provider)
    if var is None:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.getenv(var)
    if not key:
        raise RuntimeError(f"Missing {var}; add it to the .env file at the repo root")
    return key


def has_api_key(provider: str) -> bool:
    """Return True if `provider` has a non-empty API key configured."""
    var = _PROVIDER_ENV.get(provider)
    return bool(var and os.getenv(var))
