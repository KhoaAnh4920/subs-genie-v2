from mcp.server.fastmcp import FastMCP, Context
from faster_whisper import WhisperModel
import time
import sys
import logging
from typing import Optional, Literal
from utils import get_device_and_compute_type, get_app_models_dir

# Initialize FastMCP server
mcp = FastMCP("transcriber")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@mcp.tool()
def ping() -> str:
    """Ping the agent to check connectivity."""
    return "pong"


@mcp.tool()
def transcribe_audio(
    audio_path: str,
    model: str,
    ctx: Context,
    language: Optional[str] = None,
    task: Literal["transcribe", "translate"] = "transcribe",
    profile: Literal["balanced", "cjk-high-accuracy", "cjk-noisy-music"] = "balanced",
    beam_size: int = 1,
    vad_filter: bool = True,
    vad_min_silence: float = 0.3,
    chunk_length: int = 30,
    patience: float = 1.0,
    temperature_schedule: str = "0.0",
    condition_on_previous_text: bool = False,
    num_workers: int = 4,
    cpu_threads: int = 4,
) -> dict:
    """
    Transcribe audio using Faster Whisper.

    Args:
        audio_path: Absolute path to the audio file.
        model: Model name (tiny, base, small, medium, large-v2, large-v3).
        language: Language code (iso-639-1) or None for auto-detect.
        task: "transcribe" or "translate".
        profile: Tuning profile for specific scenarios (e.g. CJK).
        beam_size: Beam size for decoding.
        vad_filter: Enable VAD to filter silence.
        vad_min_silence: Minimum silence duration for VAD.
        chunk_length: Length of audio chunks.
        patience: Beam search patience factor.
        temperature_schedule: Comma-separated temperatures.
        condition_on_previous_text: Condition on previous text.
        num_workers: Number of workers.
        cpu_threads: Number of CPU threads.
    """

    ctx.info(f"Starting transcription for: {audio_path}")
    ctx.info(f"Model: {model} | Task: {task} | Profile: {profile}")

    start_time = time.time()

    # 1. Device & Compute Type Selection
    device, compute_candidates = get_device_and_compute_type()
    ctx.info(f"Device: {device} | Candidates: {compute_candidates}")

    # 2. Model Loading with Fallbacks
    models_dir = get_app_models_dir()
    whisper_model = None
    final_compute_type = None

    ctx.info("Loading model...")
    for candidate in compute_candidates:
        try:
            whisper_model = WhisperModel(
                model,
                device=device,
                compute_type=candidate,
                download_root=models_dir,
                cpu_threads=cpu_threads,
            )
            final_compute_type = candidate
            ctx.info(f"Model loaded with compute_type={candidate}")
            break
        except Exception as e:
            ctx.warning(f"Failed to load with compute_type={candidate}: {e}")

    if whisper_model is None:
        ctx.warning("Falling back to implicit compute_type...")
        try:
            whisper_model = WhisperModel(
                model,
                device=device,
                download_root=models_dir,
                cpu_threads=cpu_threads,
            )
            final_compute_type = getattr(whisper_model, "compute_type", "unknown")
            ctx.info(f"Model loaded with implicit compute_type={final_compute_type}")
        except Exception as e:
            ctx.error(f"Critical: Failed to load model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    # 3. Parameters & Profile Logic
    # Parse temperature schedule
    try:
        temps = [float(t) for t in temperature_schedule.split(",") if t.strip()]
    except ValueError:
        temps = [0.0]
    if not temps:
        temps = [0.0]

    transcribe_kwargs = {
        "task": task,
        "beam_size": max(1, beam_size),
        "patience": max(0.1, patience),
        "vad_filter": vad_filter,
        "vad_parameters": {"min_silence_duration_ms": int(vad_min_silence * 1000)},
        "chunk_length": max(1, chunk_length),
        "word_timestamps": False,  # Transcriber Agent returns raw segments
        "condition_on_previous_text": condition_on_previous_text,
        "temperature": temps if len(temps) > 1 else temps[0],
        "num_workers": max(1, num_workers),
    }

    if language:
        transcribe_kwargs["language"] = language

    # Profile Overrides
    if profile == "cjk-high-accuracy":
        transcribe_kwargs.update(
            {
                "beam_size": max(transcribe_kwargs["beam_size"], 6),
                "patience": max(transcribe_kwargs["patience"], 1.15),
                "chunk_length": max(transcribe_kwargs["chunk_length"], 60),
                "temperature": [0.0, 0.2, 0.4],
                "condition_on_previous_text": False,
            }
        )
        transcribe_kwargs["vad_parameters"]["min_silence_duration_ms"] = int(
            max(vad_min_silence, 0.2) * 1000
        )
        ctx.info("Applied cjk-high-accuracy overrides")

    elif profile == "cjk-noisy-music":
        transcribe_kwargs.update(
            {
                "beam_size": max(transcribe_kwargs["beam_size"], 5),
                "patience": max(transcribe_kwargs["patience"], 1.1),
                "chunk_length": max(transcribe_kwargs["chunk_length"], 60),
                "temperature": [0.0, 0.2, 0.4],
                "vad_filter": False,
                "condition_on_previous_text": False,
            }
        )
        ctx.info("Applied cjk-noisy-music overrides")

    # CJK Translation Tuning
    CJK_CODES = {
        "ja",
        "jpn",
        "jp",
        "ko",
        "kor",
        "kr",
        "zh",
        "zho",
        "cmn",
        "yue",
        "zh-cn",
        "zh-tw",
    }
    if task == "translate" and (language or "").lower() in CJK_CODES:
        transcribe_kwargs.update(
            {
                "beam_size": max(transcribe_kwargs["beam_size"], 6),
                "patience": max(transcribe_kwargs["patience"], 1.1),
                "temperature": [0.0, 0.2, 0.4],
                "condition_on_previous_text": False,
            }
        )
        transcribe_kwargs.pop("initial_prompt", None)
        ctx.info("Applied CJK translation tuning")

    # 4. Transcription Loop
    try:
        segments_iter, info = whisper_model.transcribe(audio_path, **transcribe_kwargs)

        all_segments = []
        duration = getattr(info, "duration", 0.0)

        for segment in segments_iter:
            # Check cancellation (MCP doesn't have native cancellation token yet,
            # relying on SIGTERM from Host, but we can check heartbeats if we implemented them)

            # Format raw segment
            seg_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
            }
            all_segments.append(seg_data)

            # Progress Logging
            if duration > 0:
                percent = int((segment.end / duration) * 100)
                ctx.report_progress(segment.end, duration)
                ctx.info(
                    f"Progress: {percent}% | {segment.start:.1f}s -> {segment.end:.1f}s"
                )

        total_time = time.time() - start_time
        ctx.info(f"Transcription complete in {total_time:.2f}s")

        return {
            "segments": all_segments,
            "meta": {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": duration,
                "total_time": total_time,
                "device": device,
                "compute_type": final_compute_type,
            },
        }

    except Exception as e:
        ctx.error(f"Transcription failed: {e}")
        import traceback

        ctx.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    mcp.run()
