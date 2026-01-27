from __future__ import annotations

import os


def default_caption_prompt_en(start_hms: str, end_hms: str) -> str:
    return (
        "You are a vision-language assistant. You will be given multiple frames from a single video clip "
        f"spanning {start_hms}–{end_hms}. "
        "Describe ONLY what is visible on screen (no audio/subtitles/metadata or outside knowledge). "
        "Write a detailed, chronological visual narrative that reads naturally as a story.\n\n"
        "Return EXACTLY one JSON object (no extra text, no markdown) with this schema:\n"
        "{\n"
        '  "clip_description": "<multi-sentence (preferably multi-paragraph) narrative of the visible events. '
        'Be concrete: who/characters (use neutral labels if unknown), clothing/colors/patterns; objects (names/quantities/states); '
        'setting/location cues (indoor/outdoor/room/setting); camera movement; lighting; legible on-screen text; '
        'entrances/exits; occlusions; and state changes across the clip. Organize from start to end; '
        'you may use lightweight hints like [start]/[mid]/[end] or [t≈HH:MM:SS] if helpful—do not invent exact timestamps. '
        'Strictly visual evidence only.>"\n'
        "}\n\n"
        "Writing requirements:\n"
        "- Visual-only: avoid speculation, backstory, or intent beyond what is visibly evident.\n"
        "- Coverage is within the prose (no lists): weave people, objects, scene, actions, camera, lighting, on-screen text, and changes into a coherent narrative.\n"
        "- Chronology: describe from the start toward the end; you may include lightweight hints like [start], [mid], [end] or [t≈00:00:05] if helpful—do not invent exact timestamps.\n"
        "- JSON constraint: return exactly one JSON object with the single key 'clip_description'; no code fences, no extra keys, no trailing text.\n"
    )


def get_caption_prompt(start_hms: str, end_hms: str, lang: str | None = None) -> str:
    lang = (lang or os.getenv("RAG_PROMPT_LANG", "zh")).lower()
    if lang == "en":
        return default_caption_prompt_en(start_hms, end_hms)
    # For now, default to English narrative to keep behavior stable across endpoints.
    return default_caption_prompt_en(start_hms, end_hms)


def default_caption_prompt(start_hms: str, end_hms: str) -> str:
    return get_caption_prompt(start_hms, end_hms)

