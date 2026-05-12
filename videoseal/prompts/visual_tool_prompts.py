from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_visual_retrieve_summary_template() -> Optional[tuple[str, Path]]:
    """Load visual_retrieve summary prompt template from env if provided."""
    env_path = (os.getenv("VISUAL_RETRIEVE_SUMMARY_PROMPT_FILE") or "").strip()
    if not env_path:
        return None
    p = Path(env_path)
    if not p.exists():
        raise FileNotFoundError(f"VISUAL_RETRIEVE_SUMMARY_PROMPT_FILE not found: {p}")
    return p.read_text(encoding="utf-8"), p


def _load_visual_inspect_system_prompt() -> Optional[str]:
    """Load optional visual_inspect system prompt from env.

    Priority (kept compatible with DynamicvideoAgentRL conventions):
      1) INSPECT_VLM_SYSTEM_PROMPT / VISUAL_INSPECT_PROMPT (raw)
      2) INSPECT_VLM_SYSTEM_PROMPT_FILE / VISUAL_INSPECT_PROMPT_FILE / VISUAL_INSPECT_SYSTEM_PROMPT_FILE (path)
    """
    sys_override = (os.getenv("INSPECT_VLM_SYSTEM_PROMPT") or os.getenv("VISUAL_INSPECT_PROMPT") or "").strip()
    if sys_override:
        return sys_override

    env_path = (
        os.getenv("INSPECT_VLM_SYSTEM_PROMPT_FILE")
        or os.getenv("VISUAL_INSPECT_PROMPT_FILE")
        or os.getenv("VISUAL_INSPECT_SYSTEM_PROMPT_FILE")
        or ""
    ).strip()
    if not env_path:
        return None
    p = Path(env_path)
    if not p.exists():
        raise FileNotFoundError(f"visual_inspect prompt file not found: {p}")

    text = p.read_text(encoding="utf-8")
    s = text.strip()
    for q in ('"""', "'''"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 6:
            s = s[len(q) : -len(q)].strip()
            break
    return s


def build_visual_retrieve_summary_prompt(
    *,
    query_text: str,
    user_question: Optional[str],
    candidates: List[Dict[str, Any]],
    max_useful: int,
    video_duration_hms: Optional[str] = None,
) -> str:
    """Prompt for summarizing visual_retrieve candidates into compact evidence for the main agent."""

    def _fmt_item(i: int, it: Dict[str, Any]) -> str:
        s = str(it.get("start_time") or "").strip()
        e = str(it.get("end_time") or "").strip()
        cap = str(it.get("caption") or "").strip().replace("\n", " ")
        return f"{i+1}. [{s}–{e}] {cap}"

    uq = (user_question or "").strip()
    qt = (query_text or "").strip()

    template = _load_visual_retrieve_summary_template()
    if template is not None:
        text, src = template
        user_block = f"User question (may include options; use them to disambiguate): {uq}\n" if uq else ""
        query_block = f"Visual retrieval query: {qt}\n" if qt else ""
        dur_line = f"Video duration: {str(video_duration_hms).strip()}\n" if (video_duration_hms and str(video_duration_hms).strip()) else ""
        candidate_lines: List[str] = []
        for i, it in enumerate(candidates[:120]):
            if isinstance(it, dict):
                s = str(it.get("start_time") or "").strip()
                e = str(it.get("end_time") or "").strip()
                cap = str(it.get("caption") or "").strip().replace("\n", " ")
                candidate_lines.append(f"{i+1}. [{s}-{e}] {cap}")
            else:
                candidate_lines.append(f"{i+1}. [--:--:-- - --:--:--] ")
        candidate_list = "\n".join(candidate_lines)
        try:
            return text.format(
                USER_QUESTION_BLOCK=user_block,
                QUERY_BLOCK=query_block,
                DUR_LINE=dur_line,
                CANDIDATE_LIST=candidate_list,
                MAX_USEFUL=str(max_useful),
            )
        except Exception as exc:
            raise RuntimeError(f"failed to format visual_retrieve_summary prompt template ({src}): {type(exc).__name__}: {exc}") from exc

    lines: List[str] = [
        "ROLE (clue finder / relevance check ONLY)",
        "- Do NOT answer the final QA and do NOT choose multiple-choice option letters.",
        "- You ONLY see candidate time windows with short captions. Use ONLY these captions.",
        "- Read the User question and extract minimal visual facts (entities, actions, relations).",
        "- Treat the Visual retrieval query as a checklist.",
        "",
        "Task:",
        "- Filter the candidate windows. Only output details for windows labeled RELATED or PARTIAL.",
        "- Ignore/Skip any window that is UNRELATED (do not list them in the output).",
        "- Separate KNOWN vs UNKNOWN facts relative to the question/options.",
        "- Propose concrete Next search cues targeting UNKNOWN facts.",
        "",
        "Output format (NO JSON, keep this order):",
        "Span relevance: <List ONLY valid spans (RELATED/PARTIAL). Format: '- [HH:MM:SS] Label: Reason'. If none are related, write 'None found'.>",
        "Evidence: <1-3 sentences; summarize KNOWN facts from the valid spans and identify UNKNOWN facts>",
        "Next search: <3-8 short phrases, derived from UNKNOWN facts>",
        f"USEFUL_SPANS:\n- HH:MM:SS–HH:MM:SS\n- ... (List strictly the timestamps of the RELATED/PARTIAL spans identified above)",
        "",
        "Notes:",
        "- If no spans are clearly related, state this in Evidence, and leave USEFUL_SPANS empty or list 1 weak lead.",
        "- Do NOT produce a numbered list of all 30 items. Only show the hits.",
        "",
    ]
    if uq:
        lines.append(f"User question (may include options; use them to disambiguate): {uq}")
    if qt:
        lines.append(f"Visual retrieval query: {qt}")
    lines.append("Candidate windows:")

    for i, it in enumerate(candidates[:120]):
        if isinstance(it, dict):
            lines.append(_fmt_item(i, it))
        else:
            lines.append(f"{i+1}. [--:--:--–--:--:--] ")
    return "\n".join(lines)


def build_visual_inspect_prompt(spans_label: str, qtext: str, *, context: Optional[str] = None, prompt_type: int = 0) -> str:
    """Prompt for multi-span visual_inspect tool (VLM, free-form answer).

    Kept aligned with DynamicvideoAgentRL defaults.
    """
    sys_prompt = _load_visual_inspect_system_prompt()
    if sys_prompt:
        ctx = (str(context).strip() if (context and str(context).strip()) else "")
        ctx_block = f"Context:\n{ctx}\n\n" if ctx else ""
        return (
            f"{sys_prompt.strip()}\n\n"
            f"Spans: {spans_label}\n\n"
            f"{ctx_block}"
            f"Question(s):\n{qtext}\n"
        )

    ctx_block = ("\nContext: " + str(context).strip()) if (context and str(context).strip()) else ""
    return (
        "You will be shown frames sampled across multiple time windows of a video. "
        f"Spans: {spans_label}." + ctx_block + "\n"
        "Use the frames together with the optional Context (which summarizes prior searches) to answer the question. "
        "Reason explicitly about how the visual evidence matches each part of the question and any clues in Context.\n"
        "Important: You ONLY see frames from the spans listed above. If a required element is not visible or cannot be inferred here, "
        "that usually means the answer is outside these spans and the retriever should search elsewhere in the video. "
        "Only treat these spans as FIXED ground-truth if the question explicitly asks about this exact time range "
        "(e.g., it mentions the same timestamps or says 'in this segment'). In that fixed-range case, you MUST choose the best-supported option from these frames, "
        "even if your confidence is below 0.95, and you must NOT output SEARCH_MORE.\n"
        "Only choose option letters if the frames (with help from Context) clearly support **all** required elements "
        "(entities, actions, attributes, relations, and any time conditions). "
        "If this is NOT a fixed-range question and any required element cannot be verified or reliably inferred from these frames, do NOT choose an option.\n"
        "Always output a numeric confidence in [0.00, 1.00]. If this is NOT a fixed-range question and confidence is below 0.95, output Answer: SEARCH_MORE.\n"
        "When outputting SEARCH_MORE, briefly state which specific elements of the question are still unverified in these spans "
        "and suggest 1–3 concrete next searches as **query phrases / what to look for**. "
        "Do NOT include any explicit timestamps, HH:MM:SS ranges, or relative time directions (e.g., earlier/later). "
        "Do NOT recommend re-checking the same spans unless the question itself requires it. "
        "Finally, add a short Reminder line telling the central LLM that the NEXT tool call should NOT be visual_inspect, "
        "and should instead use other retrieval tools to find new spans.\n\n"
        "When citing visual evidence in your answer, reference exact timestamps in HH:MM:SS (or ranges like HH:MM:SS–HH:MM:SS) from the provided spans; do NOT use \"frame N\" or image indices.\n\n"
        f"Question(s):\n{qtext}\n\n"
        "OUTPUT FORMAT\n"
        "- Start with a single line: Answer: <one or more option letters such as \"A\" or \"A,C\", OR \"SEARCH_MORE\">.\n"
        "- Then provide Evidence: <1–3 sentences grounded in the frames; include HH:MM:SS timestamps rather than \"frame N\" when referencing specific visual cues>.\n"
        "- Then Confidence: <0.00–1.00, two decimals>.\n"
        "- If Answer is SEARCH_MORE, add three more lines:\n"
        "  Missing: <bullet-like list inside one line of what is not visible / still unclear>.\n"
        "  Next search: <concrete query phrases / what to search for next, WITHOUT any timestamps>.\n\n"
        "  Reminder: <one sentence: do NOT call visual_inspect next; use other tools to search new spans>.\n\n"
        "If you output option letters in a non-fixed-range question, your confidence MUST be at least 0.95 and your evidence must cover every element of the question."
    )


def build_visual_inspect_last_step_mcq_prompt(spans_label: str, qtext: str, *, context: Optional[str] = None) -> str:
    """MCQ answer-only prompt for last-step visual_inspect (force a choice letter)."""
    ctx_block = ("\nContext:\n" + str(context).strip()) if (context and str(context).strip()) else ""
    base = (
        "Select the best answer to the following multiple-choice question based on the video frames.\n"
        "Respond with only the letter of the correct option (e.g., A). Output nothing else.\n"
        f"\nSpans: {spans_label}{ctx_block}\n\n"
        f"Question:\n{qtext}\n\n"
        "The best answer is:\n"
    )
    sys_prompt = _load_visual_inspect_system_prompt()
    if sys_prompt:
        return f"{sys_prompt.strip()}\n\n{base}"
    return base
