from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def gen_run_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tail = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    return f"{now}-{tail}"


@dataclass
class Step:
    chat_prompt: str = ""
    model_response: str = ""
    thinking: str | None = None
    action: Dict[str, Any] | None = None
    observation: Dict[str, Any] | None = None
    usage: Dict[str, Any] | None = None


@dataclass
class Trajectory:
    system_prompt: str = ""
    tools_schema: str = ""
    question: str = ""
    time_reference: str | None = None
    uid: str | None = None
    groundtruth: str | None = None
    video_id: str = ""
    steps: List[Step] = field(default_factory=list)
    answer: str | None = None
    run_id: str = field(default_factory=gen_run_id)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    elapsed_sec: float | None = None
    note: str | None = None

