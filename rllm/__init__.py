"""rLLM: Reinforcement Learning with Language Models."""

__all__ = ["BaseAgent", "Action", "Step", "Trajectory", "Episode"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from . import agents as _agents
    except Exception as exc:
        raise ImportError(
            "rllm agent classes require optional dependencies (e.g., torch). "
            "Install them or import submodules that do not require those deps."
        ) from exc
    return getattr(_agents, name)
