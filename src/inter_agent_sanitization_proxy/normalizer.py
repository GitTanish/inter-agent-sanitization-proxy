from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from .models import Message, MessageContent

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_BIDI_CONTROLS = dict.fromkeys(
    map(
        ord,
        "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069",
    ),
    None,
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v]+")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    original: Message
    content: MessageContent
    metadata: dict[str, Any]
    content_format: Literal["text", "structured"]


class Normalizer:
    def normalize_message(self, message: Message) -> NormalizedMessage:
        if not isinstance(message, Message):
            raise TypeError("message must be a Message")

        content = self._normalize_value(message.content)
        metadata = self._normalize_value(message.metadata)
        content_format: Literal["text", "structured"] = (
            "text" if isinstance(message.content, str) else "structured"
        )

        return NormalizedMessage(
            original=message,
            content=content,
            metadata=metadata,
            content_format=content_format,
        )

    def normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = unicodedata.normalize("NFKC", text)
        text = text.translate(_ZERO_WIDTH).translate(_BIDI_CONTROLS)
        text = _CONTROL_CHARS.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _HORIZONTAL_WHITESPACE.sub(" ", text)
        text = _TRAILING_WHITESPACE.sub("", text)
        text = _EXCESS_NEWLINES.sub("\n\n", text)
        return text.strip()

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.normalize_text(value)
        if isinstance(value, dict):
            return {
                key: self._normalize_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_value(item) for item in value]
        return value
