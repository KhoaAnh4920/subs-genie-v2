import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Union

_PUNCT_END = re.compile(r"[.!?…]$")
_SMALL_WORD = re.compile(
    r"^(a|an|the|and|or|but|to|of|in|on|at|for|with|your|my|his|her|their|our|is|am|are|was|were|it|that|this|there|here)\b",
    re.I,
)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _cps(text: str, duration_s: float) -> float:
    chars = len(text.replace(" ", ""))
    return chars / max(duration_s, 1e-6)


def _light_grammar(text: str, prev_ended_sentence: bool) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([(\[{])\s+", r"\1", t)
    t = re.sub(r"\s+([)\]}])", r"\1", t)
    t = t.replace(" i ", " I ")
    t = re.sub(r"^i\b", "I", t)
    if prev_ended_sentence and t and t[0].isalpha():
        t = t[0].upper() + t[1:]
    return t


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def post_process_segments(
    segments: List[Dict[str, Any]],
    grammar_style: bool = False,
    max_chars_per_line: int = 42,
    min_duration: float = 1.0,
    max_duration: float = 6.0,
) -> List[Dict[str, Any]]:
    """
    Apply advanced post-processing to subtitles:
    1. Merge broken sentences.
    2. Dedupe near-duplicates.
    3. Enforce CPL (Chars Per Line), CPS (Chars Per Second), and duration constraints.
    """

    # 1. Merge broken sentences
    merged = []
    for seg in segments:
        if merged:
            A = merged[-1]
            gap = max(0.0, seg["start"] - A["end"])
            A_text, B_text = A["text"].strip(), seg["text"].strip()

            # Heuristic: Merge if A doesn't end with punctuation AND
            # (B starts with small word OR A is very short) AND gap is small
            if (
                not _PUNCT_END.search(A_text)
                and (_SMALL_WORD.match(B_text) or len(A_text.split()) <= 3)
                and gap < 0.5
            ):
                new_text = (A_text + " " + B_text).strip()
                new_dur = seg["end"] - A["start"]
                # Only merge if resulting segment isn't too fast or too long
                if new_dur <= max_duration and _cps(new_text, new_dur) <= 17.0:
                    A["text"] = new_text
                    A["end"] = seg["end"]
                    continue
        merged.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})

    # 2. Dedupe near-duplicates within 1s window (Fixes Whisper hallucinations)
    deduped = []
    i = 0
    while i < len(merged):
        cur = merged[i]
        cur_norm = _norm_text(cur["text"])
        j = i + 1
        while j < len(merged):
            nxt = merged[j]
            # If segments are too far apart, stop checking for duplicates
            if nxt["start"] - cur["end"] > 1.0:
                break
            nxt_norm = _norm_text(nxt["text"])

            # If >90% similar, keep the longer one (or merge time range)
            if _similarity(cur_norm, nxt_norm) >= 0.9:
                keep_text = (
                    cur["text"] if len(cur_norm) >= len(nxt_norm) else nxt["text"]
                )
                cur["text"] = keep_text
                cur["end"] = max(cur["end"], nxt["end"])
                j += 1
            else:
                break
        deduped.append(cur)
        i = j

    # 3. Enforce limits: 2 lines, Max CPL, Duration Clamping
    final = []
    sentence_end = True
    for seg in deduped:
        text = seg["text"]
        dur = seg["end"] - seg["start"]

        t = (
            _light_grammar(text, prev_ended_sentence=sentence_end)
            if grammar_style
            else text.strip()
        )
        sentence_end = bool(_PUNCT_END.search(t))

        # Line Splitting
        words = t.split()
        line1, line2 = [], []
        for w in words:
            if len(" ".join(line1 + [w])) <= max_chars_per_line:
                line1.append(w)
            else:
                line2.append(w)
        wrapped = (
            " ".join(line1) if not line2 else " ".join(line1) + "\n" + " ".join(line2)
        )

        # Smart Split: If fast CPS and long duration, split into two segments
        if _cps(t, dur) > 17.0 and dur >= 3.0 and len(words) >= 6:
            mid = seg["start"] + dur / 2
            half = len(words) // 2
            t1 = " ".join(words[:half]).strip()
            t2 = " ".join(words[half:]).strip()
            if t1:
                final.append({"start": seg["start"], "end": mid, "text": t1})
            if t2:
                final.append({"start": mid, "end": seg["end"], "text": t2})
        else:
            # Duration Clamping
            if dur < min_duration:
                seg["end"] = seg["start"] + min_duration
            elif dur > max_duration:
                seg["end"] = seg["start"] + max_duration

            final.append({"start": seg["start"], "end": seg["end"], "text": wrapped})

    return final
