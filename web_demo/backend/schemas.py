from typing import Literal, Optional
from pydantic import BaseModel


Mode = Literal["normal", "rag", "mcp"]


class ClientMessage(BaseModel):
    type: Literal["user_message", "set_mode", "reset", "ping"]
    text: Optional[str] = None
    mode: Optional[Mode] = None


class TranscribeResponse(BaseModel):
    text: str
