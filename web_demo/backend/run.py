import os

# Avoid hidden corruption in nested-thread inference paths.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import uvicorn


def main() -> None:
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
