from __future__ import annotations

from typing import Dict, Type

from videoseal.tools.base import Tool
from videoseal.tools.visual_tools import VisualInspectAliasTool, VisualRetrieveAliasTool


def build_tool_map() -> Dict[str, Type[Tool]]:
    return {
        "visual_retrieve": VisualRetrieveAliasTool,
        "visual_inspect": VisualInspectAliasTool,
    }

