from __future__ import annotations

import re
from typing import Dict, Optional, Tuple, List


def parse_choice_letter(answer_text: str) -> Optional[str]:
    if not answer_text:
        return None
    raw = str(answer_text)
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        raw = m.group(1)
    else:
        m = re.search(r"<final>\s*(.*?)\s*</final>", raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            raw = m.group(1)
        else:
            m = re.search(r"<(?:answer|final)>\s*([A-Za-z])(?=[^A-Za-z]|$)", raw, flags=re.DOTALL | re.IGNORECASE)
            if m:
                raw = m.group(1)

    table = str.maketrans({"（": "(", "）": ")", "【": "[", "】": "]", "：": ":", "，": ","})
    text = str(raw).strip().translate(table)

    m = re.search(r"(?:\*\*)?\s*[\(\[]\s*([A-Za-z])\s*[\)\]]\s*(?:\*\*)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(
        r"(?:选|答案|选项|option|answer|choice|selected|select|choose)\s*(?:is)?\s*(?:[:：])?\s*([A-Za-z])(?=\b|[\)\]\s,.;:，。；：]|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    s = text.strip()
    m = re.fullmatch(r"([A-Za-z])(?:\s*[\.)])?\s*", s)
    if m:
        return m.group(1).upper()
    return None


def _normalize_text(s: str) -> str:
    table = str.maketrans({"（": "(", "）": ")", "【": "[", "】": "]", "：": ":", "，": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})
    s = str(s).translate(table)
    s = s.replace("&", " and ").replace("＋", "+").replace("、", " and ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_options(question_text: str) -> Dict[str, str]:
    q = _normalize_text(question_text)
    spans: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"\(([A-Za-z])\)|\[([A-Za-z])\]|\b([A-Za-z])[\)\.:：、]\s*", q):
        letter = None
        for gi in range(1, 4):
            if m.group(gi):
                letter = m.group(gi)
                break
        if not letter:
            continue
        spans.append((letter.upper(), m.start(), m.end()))
    if len(spans) < 2:
        for m in re.finditer(r"(?m)^(?:\(|\[)?([A-Za-z])(?:\)|\])?[\)\.:：、]\s+", q):
            spans.append((m.group(1).upper(), m.start(), m.end()))
    mapping: Dict[str, str] = {}
    spans.sort(key=lambda x: x[1])
    for i, (let, _s, epos) in enumerate(spans):
        nxt = spans[i + 1][1] if i + 1 < len(spans) else len(q)
        text = q[epos:nxt].strip().strip(" -—:\\n\\t ")
        if text:
            mapping[let] = text
    return mapping


def parse_choice_letter_smart(answer_text: str, question_text: Optional[str] = None) -> Optional[str]:
    letter = parse_choice_letter(answer_text)
    if letter:
        return letter
    if not question_text:
        return None

    ans = _normalize_text(str(answer_text or ""))
    if not ans:
        return None
    opts = _extract_options(question_text)
    if not opts:
        return None

    best: Tuple[str, int] | None = None
    for let, txt in opts.items():
        t = _normalize_text(txt)
        if not t:
            continue
        if t.lower() in ans.lower():
            score = len(t)
            if best is None or score > best[1]:
                best = (let, score)
    return best[0] if best else None
