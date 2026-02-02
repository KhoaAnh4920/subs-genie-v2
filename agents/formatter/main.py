from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Literal
from formatting import post_process_segments, format_timestamp, format_timestamp_vtt
import logging

# Initialize FastMCP server
mcp = FastMCP("formatter")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@mcp.tool()
def format_subtitles(
    segments: List[Dict[str, Any]],
    format: Literal["srt", "vtt", "txt"] = "srt",
    grammar_style: bool = False,
    max_chars_per_line: int = 42,
    min_duration: float = 1.0,
    max_duration: float = 6.0,
) -> str:
    """
    Format raw transcription segments into SRT, VTT, or TXT.

    Args:
        segments: List of raw segments [{"start": 0.0, "end": 2.0, "text": "..."}]
        format: Output format (srt, vtt, txt)
        grammar_style: Apply light grammar fixes (capitalization, punctuation).
        max_chars_per_line: Maximum characters per line (default 42).
        min_duration: Minimum duration for a segment in seconds.
        max_duration: Maximum duration for a segment in seconds.
    """
    try:
        logger.info(f"Formatting {len(segments)} segments to {format.upper()}")

        # 1. Post-process segments (Merge, Dedupe, Split)
        processed = post_process_segments(
            segments,
            grammar_style=grammar_style,
            max_chars_per_line=max_chars_per_line,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        # 2. Generate Output String
        output = ""

        if format == "srt":
            for i, seg in enumerate(processed, 1):
                start = format_timestamp(seg["start"])
                end = format_timestamp(seg["end"])
                text = seg["text"].strip()
                output += f"{i}\n{start} --> {end}\n{text}\n\n"

        elif format == "vtt":
            output = "WEBVTT\n\n"
            for seg in processed:
                start = format_timestamp_vtt(seg["start"])
                end = format_timestamp_vtt(seg["end"])
                text = seg["text"].strip()
                output += f"{start} --> {end}\n{text}\n\n"

        else:  # txt
            output = "\n".join([s["text"].strip() for s in processed])

        return output

    except Exception as e:
        logger.error(f"Formatting failed: {e}")
        raise RuntimeError(f"Formatting failed: {e}")


if __name__ == "__main__":
    mcp.run()
