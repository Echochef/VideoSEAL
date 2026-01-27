from __future__ import annotations

import importlib
import warnings

from rllm.tools.registry import ToolRegistry

tool_registry = ToolRegistry()

_DEFAULT_TOOL_SPECS: dict[str, tuple[str, str]] = {
    "python": ("rllm.tools.code_tools", "PythonInterpreter"),
    "google_search": ("rllm.tools.web_tools", "GoogleSearchTool"),
    "firecrawl": ("rllm.tools.web_tools", "FirecrawlTool"),
    "tavily-extract": ("rllm.tools.web_tools", "TavilyExtractTool"),
    "tavily-search": ("rllm.tools.web_tools", "TavilySearchTool"),
    # Video tools (LVU-style)
    "visual_retrieve": ("rllm.tools.video_tools", "VisualRetrieveTool"),
    "visual_inspect": ("rllm.tools.video_tools", "VisualInspectTool"),
}

DEFAULT_TOOLS: dict[str, type] = {}
_defaults_registered = False


def _register_default_tools() -> None:
    global _defaults_registered
    if _defaults_registered:
        return
    for name, (module_path, class_name) in _DEFAULT_TOOL_SPECS.items():
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            warnings.warn(
                f"Skipping tool '{name}': failed to import {module_path} ({exc})",
                stacklevel=2,
            )
            continue
        tool_cls = getattr(module, class_name, None)
        if tool_cls is None:
            raise ImportError(f"Tool class {class_name!r} not found in {module_path}")
        DEFAULT_TOOLS[name] = tool_cls
        tool_registry.register(name, tool_cls)
    _defaults_registered = True


def __getattr__(name: str):
    if name == "tool_registry":
        _register_default_tools()
        return tool_registry
    if name == "ToolRegistry":
        return ToolRegistry
    for _, (module_path, class_name) in _DEFAULT_TOOL_SPECS.items():
        if name == class_name:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _ensure_tools_registered() -> None:
    _register_default_tools()


_register_default_tools()

__all__ = [
    "ToolRegistry",
    "tool_registry",
    "DEFAULT_TOOLS",
    "_ensure_tools_registered",
]
__all__.extend([cls for _, cls in _DEFAULT_TOOL_SPECS.values()])
