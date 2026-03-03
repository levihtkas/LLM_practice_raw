from __future__ import annotations

from typing import Iterable


class Preprocessor:
    """Basic text preprocessor utility."""

    def __init__(self, lowercase: bool = True, strip: bool = True) -> None:
        self.lowercase = lowercase
        self.strip = strip

    def transform(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        value = text
        if self.strip:
            value = value.strip()
        if self.lowercase:
            value = value.lower()
        return value

    def batch_transform(self, texts: Iterable[str]) -> list[str]:
        return [self.transform(text) for text in texts]
