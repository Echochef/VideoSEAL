from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolOutput:
    name: str
    output: Any | None = None
    error: str | None = None
    metadata: dict | None = None

    def __str__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        if self.output is None:
            return ""
        if isinstance(self.output, (list, dict)):
            return json.dumps(self.output, ensure_ascii=False)
        return str(self.output)

    def to_string(self) -> str:
        return str(self)


class Tool:
    def __init__(self, name: str | None = None, description: str | None = None):
        self.name = name or self.__class__.__name__
        self.description = description or ""

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def forward(self, *args, **kwargs) -> ToolOutput:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> ToolOutput:
        return self.forward(*args, **kwargs)

