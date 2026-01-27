from __future__ import annotations


def build_full_story_prompt(timeline_records: str) -> str:
    return (
        "You are given a chronological timeline of a video composed of many time windows, each in the form\n"
        "[HH:MM:SS–HH:MM:SS] followed by concise notes that may combine visual scene information and short on‑screen text.\n\n"
        "TASK\n"
        "Write ONE long, comprehensive narrative that reconstructs the full story of the entire video from start to end.\n"
        "Maximize information retention (avoid aggressive compression). Keep the chronology strict and preserve key details,\n"
        "including visible entities, actions, scene changes, and clearly legible on‑screen text. You may inline time anchors\n"
        "like [HH:MM:SS–HH:MM:SS] when helpful for later localization. Do not invent facts beyond what the records support.\n\n"
        "GUIDELINES\n"
        "1) Maintain temporal order; describe the flow of events as they unfold.\n"
        "2) Integrate visual descriptions and textual cues naturally; quote short critical text exactly.\n"
        "3) Prefer concrete nouns/verbs over vague phrasing; keep proper names if provided.\n"
        "4) If adjacent windows clearly refer to one continuous scene, you may merge them in prose (but do not omit details).\n"
        "5) If some windows are uncertain/ambiguous, note it briefly without speculating.\n\n"
        "OUTPUT\n"
        "Return a single long English paragraph (or a few long paragraphs) that covers the entire video. Do not output JSON.\n\n"
        "TIMELINE WINDOWS\n"
        f"{timeline_records}"
    )


def build_chunk_scene_fusion_prompt_en(time_range: str, window_notes: str) -> str:
    return (
        f"[Chunk Time] {time_range}\n"
        f"[Merged Notes]\n{window_notes}\n\n"
        "TASK\n"
        "You are a film storyline editor. Using only the material above (already merged visual scene information and on‑screen text/captions),\n"
        "write a coherent narrative for this chunk that prioritizes storyline and temporal logic. Pepper the prose with precise time anchors\n"
        "referencing salient sub‑windows from the notes. Use bracketed anchors like [HH:MM:SS–HH:MM:SS] or [HH:MM:SS] inserted inline right\n"
        "before or after the action they refer to.\n\n"
        "REQUIREMENTS\n"
        "- Strict chronology: who is where, doing/saying what, and what changes as an immediate result.\n"
        "- Ground every beat in the provided captions and visual cues; do not invent names, settings, or events beyond the material.\n"
        "- Integrate visual and textual cues naturally; avoid meta phrases like “the subtitle shows…”. Quote very short key phrases (≤10 words) only to anchor critical moments.\n"
        "- Include 2–4 inline time anchors tied to key beats (entrances/exits, reveals, decisions, scene turns).\n"
        "- Focus on cause→effect links, turning points, evolving goals/obstacles, and emotional shifts (e.g., calm → alarmed → resolved).\n"
        "- Keep names/roles consistent; if names are missing, use stable role descriptors (e.g., “the man in the robe”, “the woman at the door”).\n"
        "- Use present tense, third person. Avoid camera/technical terms.\n"
        "- Write as much as needed for clarity and completeness; never paste raw lines from [Merged Notes].\n\n"
        "OUTPUT\n"
        "Return a single English paragraph (not a list) that already contains the inline time anchors."
    )


def build_global_storyline_prompt_en(chunk_summaries: str) -> str:
    return (
        "[Chunk Narratives]\n"
        f"{chunk_summaries}\n\n"
        "TASK\n"
        "You are a senior storyline editor. Merge all chunk narratives into a complete, storyline‑first film arc.\n"
        "Maintain strict chronology; preserve and integrate scene and dialogue cues; for each chunk, keep its [HH:MM:SS–HH:MM:SS] header and weave\n"
        "1–3 inline time anchors ([HH:MM:SS–HH:MM:SS] or [HH:MM:SS]) inside the sentences to mark key beats.\n"
        "Resolve pronouns and keep entity names/roles consistent across chunks. Highlight goals, conflicts, obstacles, turning points, and interim outcomes as they evolve.\n"
        "Ground everything in the provided material only; do not invent new names, settings, or events.\n\n"
        "OUTPUT FORMAT\n"
        "- Per‑chunk (in time order):\n"
        "  [HH:MM:SS–HH:MM:SS] Segment: <2–4 sentences; include 1–3 inline time anchors tied to pivotal actions/reveals/turns>\n"
        "  … (list all chunks in order)\n"
        "- Full Film Summary: <a continuous narrative capturing the main arc—goals, conflicts, escalations, turning points, and resolution; use only provided material>\n\n"
        "NOTES\n"
        "- Do not echo raw note lines; return only authored narrative.\n"
        "- You may quote very short key phrases (≤10 words) to anchor critical moments.\n"
        "- Avoid bullet points and camera/technical terminology; keep the prose coherent and cinematic.\n"
        "- Use present tense, third person. If chunks contain minor contradictions, prefer the clearest chronology and caption‑anchored evidence.\n"
    )

