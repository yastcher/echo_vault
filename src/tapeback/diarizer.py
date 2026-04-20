import contextlib
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

from tapeback import const
from tapeback.channel import classify_segment_by_channel, load_stereo_channels
from tapeback.models import DiarizationSegment, Segment
from tapeback.settings import Settings
from tapeback.speaker_merge import merge_similar_speakers

# pyannote segmentation + embedding models need ~1-1.5 GB VRAM during inference
DIARIZATION_VRAM_MIN_MIB = 1500

__all__ = [
    "Diarizer",
    "assign_speakers",
    "consolidate_segments",
    "diarization_available",
    "merge_channel_segments",
    "merge_similar_speakers",
]


def _get_free_vram_mib() -> int | None:
    """Get free GPU VRAM in MiB via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def diarization_available() -> bool:
    """Check if pyannote-audio is installed."""
    try:
        import pyannote.audio  # noqa: F401

        return True
    except ImportError:
        return False


def _unwrap_diarization(result: Any) -> Any:
    """Extract Annotation from pyannote output.

    pyannote 4.x returns DiarizeOutput (with .speaker_diarization),
    older versions return Annotation directly (with .itertracks).
    """
    if hasattr(result, "itertracks"):
        return result
    return result.speaker_diarization


class Diarizer:
    def __init__(self, settings: Settings) -> None:
        """Initialize pyannote pipeline.

        Raises RuntimeError if hf_token is empty.
        Falls back to CPU if CUDA is not available.
        """
        hf_token = settings.hf_token.get_secret_value()
        if not hf_token:
            raise RuntimeError(
                "HuggingFace token required for diarization. "
                "Set TAPEBACK_HF_TOKEN in your .env file."
            )

        from pyannote.audio import Pipeline

        self._settings = settings
        pipeline = Pipeline.from_pretrained(
            const.PYANNOTE_MODEL,
            token=hf_token,
        )
        if pipeline is None:
            raise RuntimeError("Failed to load pyannote diarization pipeline")
        self._pipeline = pipeline

        if settings.clustering_threshold is not None:
            params = self._pipeline.parameters(instantiated=True)
            params["clustering"]["threshold"] = settings.clustering_threshold
            self._pipeline.instantiate(params)

        if settings.device == "cuda":
            free_mib = _get_free_vram_mib()
            if free_mib is not None and free_mib < DIARIZATION_VRAM_MIN_MIB:
                print(
                    f"Warning: Not enough VRAM for diarization "
                    f"({free_mib} MiB free < {DIARIZATION_VRAM_MIN_MIB} MiB), using CPU",
                    file=sys.stderr,
                )
            else:
                try:
                    import torch

                    self._pipeline.to(torch.device("cuda"))
                except RuntimeError:
                    print(
                        "Warning: CUDA not available for diarization, using CPU",
                        file=sys.stderr,
                    )

    def _run_pipeline(self, audio_path: Path) -> Any:
        """Run pyannote pipeline with optional max_speakers."""
        if self._settings.max_speakers is not None:
            return self._pipeline(audio_path, max_speakers=self._settings.max_speakers)
        return self._pipeline(audio_path)

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        """Run diarization on audio file.

        Returns list of DiarizationSegment sorted by start time.
        Falls back to CPU if CUDA runs out of memory during inference.
        """
        try:
            diarization = self._run_pipeline(audio_path)
        except RuntimeError as exc:
            if "CUDA" not in str(exc) and "out of memory" not in str(exc):
                raise
            print(
                f"Warning: CUDA out of memory during diarization, falling back to CPU: {exc}",
                file=sys.stderr,
            )
            self._fallback_to_cpu()
            diarization = self._run_pipeline(audio_path)

        # pyannote 4.x returns DiarizeOutput wrapping Annotation
        annotation = _unwrap_diarization(diarization)

        segments: list[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end,
                )
            )

        return segments

    def _fallback_to_cpu(self) -> None:
        """Move pipeline to CPU and free CUDA memory."""
        import torch

        self._pipeline.to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def merge_channel_segments(
    mic_segments: list[Segment],
    monitor_segments: list[Segment],
) -> list[Segment]:
    """Merge segments from both channels, sorted by start time.

    After sorting, consecutive segments from the same speaker are consolidated
    into a single segment.  Overlapping segments from different speakers are
    kept separate.
    """
    merged = sorted(mic_segments + monitor_segments, key=lambda s: s.start)
    return consolidate_segments(merged)


def consolidate_segments(segments: list[Segment]) -> list[Segment]:
    """Merge consecutive segments from the same speaker into one.

    Handles both adjacent and overlapping segments.  Preserves word lists
    by concatenation.
    """
    if not segments:
        return []

    result: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        if prev.speaker and prev.speaker == seg.speaker:
            # Merge: extend previous segment
            words = None
            if prev.words and seg.words:
                words = prev.words + seg.words
            elif prev.words or seg.words:
                words = prev.words or seg.words
            result[-1] = Segment(
                start=prev.start,
                end=max(prev.end, seg.end),
                text=prev.text + " " + seg.text,
                words=words,
                speaker=prev.speaker,
            )
        else:
            result.append(seg)

    return result


def _find_speaker_for_time(
    start: float,
    end: float,
    diarization_segments: list[DiarizationSegment],
) -> str | None:
    """Find the pyannote speaker with most overlap for a time range."""
    best_speaker = None
    best_overlap = 0.0

    for dseg in diarization_segments:
        overlap = max(0.0, min(end, dseg.end) - max(start, dseg.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = dseg.speaker

    # No overlap — find nearest segment
    if best_speaker is None:
        mid = (start + end) / 2
        min_dist = float("inf")
        for dseg in diarization_segments:
            dist = min(abs(mid - dseg.start), abs(mid - dseg.end))
            if dist < min_dist:
                min_dist = dist
                best_speaker = dseg.speaker

    return best_speaker


def _resegment_by_words(
    segment: Segment,
    diarization_segments: list[DiarizationSegment],
) -> list[tuple[Segment, str | None]]:
    """Split a segment into sub-segments at diarization speaker boundaries.

    Each word is assigned to its pyannote speaker.  Consecutive words from the
    same speaker are grouped into one sub-segment.  Returns list of
    (sub_segment, pyannote_speaker) tuples.
    """
    if not segment.words:
        speaker = _find_speaker_for_time(segment.start, segment.end, diarization_segments)
        return [(segment, speaker)]

    # Assign each word to a pyannote speaker
    word_speakers: list[tuple[str | None, int]] = []
    for i, word in enumerate(segment.words):
        speaker = _find_speaker_for_time(word.start, word.end, diarization_segments)
        word_speakers.append((speaker, i))

    # Group consecutive same-speaker words into sub-segments
    result: list[tuple[Segment, str | None]] = []
    group_start = 0

    for i in range(1, len(word_speakers) + 1):
        if i < len(word_speakers) and word_speakers[i][0] == word_speakers[group_start][0]:
            continue

        # Flush group [group_start, i)
        group_words = segment.words[group_start:i]
        speaker = word_speakers[group_start][0]
        text = "".join(w.word for w in group_words).strip()
        if text:
            result.append(
                (
                    Segment(
                        start=group_words[0].start,
                        end=group_words[-1].end,
                        text=text,
                        words=group_words,
                    ),
                    speaker,
                )
            )
        group_start = i

    return result if result else [(segment, None)]


def _resolve_speaker_label(
    pyannote_speaker: str,
    channel: str | None,
    speaker_order: list[str],
    user_speaker: str | None,
) -> str:
    """Pick a display label for a pyannote speaker, honouring channel override."""
    if channel == "mic":
        return const.SPEAKER_YOU
    non_user = [s for s in speaker_order if s != user_speaker]
    if channel is None and user_speaker and pyannote_speaker == user_speaker:
        return const.SPEAKER_YOU
    if pyannote_speaker in non_user:
        idx = non_user.index(pyannote_speaker) + 1
    else:
        idx = len(non_user) + 1
    return const.SPEAKER_LABEL_FMT.format(idx)


def _classify_sub_segment(
    sub_seg: Segment,
    stereo_data: tuple[np.ndarray, np.ndarray, int] | None,
) -> str | None:
    """Classify a sub-segment by channel energy, or None if no stereo data."""
    if stereo_data is None:
        return None
    mic, monitor, sr = stereo_data
    return classify_segment_by_channel(sub_seg.start, sub_seg.end, mic, monitor, sr)


def _label_from_channel_only(channel: str | None) -> str | None:
    """Fallback label when pyannote couldn't assign a speaker for a sub-segment."""
    if channel == "mic":
        return const.SPEAKER_YOU
    if channel == "monitor":
        return const.SPEAKER_LABEL_FMT.format(1)
    return None


def assign_speakers(
    segments: list[Segment],
    diarization_segments: list[DiarizationSegment],
    user_speaker: str | None = None,
    stereo_wav: Path | None = None,
) -> list[Segment]:
    """Assign speaker labels to each Segment based on channel energy + pyannote.

    For segments with word timestamps, splits at diarization speaker boundaries
    so that words from different speakers become separate segments.

    When stereo_wav is provided, per-segment channel energy determines
    "You" (mic-dominant) vs "Others" (monitor-dominant). Ambiguous segments
    fall back to pyannote-based assignment via user_speaker.

    Speaker naming:
    - mic-dominant or user_speaker -> "You"
    - Others -> "Speaker 1", "Speaker 2", ... (numbered by appearance order)

    Does NOT mutate input segments — creates new ones.
    """
    if not diarization_segments:
        return list(segments)

    stereo_data: tuple[np.ndarray, np.ndarray, int] | None = None
    if stereo_wav is not None:
        with contextlib.suppress(ValueError, wave.Error):
            stereo_data = load_stereo_channels(stereo_wav)

    speaker_order: list[str] = []
    result: list[Segment] = []

    for seg in segments:
        for sub_seg, pyannote_speaker in _resegment_by_words(seg, diarization_segments):
            if pyannote_speaker and pyannote_speaker not in speaker_order:
                speaker_order.append(pyannote_speaker)

            channel = _classify_sub_segment(sub_seg, stereo_data)
            if pyannote_speaker:
                label: str | None = _resolve_speaker_label(
                    pyannote_speaker, channel, speaker_order, user_speaker
                )
            else:
                label = _label_from_channel_only(channel)

            result.append(
                Segment(
                    start=sub_seg.start,
                    end=sub_seg.end,
                    text=sub_seg.text,
                    words=sub_seg.words,
                    speaker=label,
                )
            )

    return result
