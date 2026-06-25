"""Single source of truth for backend LLM API keys.

Keys must be exported into the process environment. Callers use
`get_api_key` / `has_api_key`.
"""

import os

_PROVIDER_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def get_api_key(provider: str) -> str:
    """Return the API key for `provider`, raising if it is unset."""
    var = get_api_key_env_var(provider)
    key = os.getenv(var)
    if not key:
        raise RuntimeError(f'Missing {var}; set it with `export {var}="..."`')
    return key


def get_api_key_env_var(provider: str) -> str:
    """Return the environment variable name for `provider`."""
    var = _PROVIDER_ENV.get(provider)
    if var is None:
        raise ValueError(f"Unknown provider: {provider}")
    return var


def has_api_key(provider: str) -> bool:
    """Return True if `provider` has a non-empty API key configured."""
    var = _PROVIDER_ENV.get(provider)
    return bool(var and os.getenv(var))
