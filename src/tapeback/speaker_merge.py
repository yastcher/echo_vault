"""Spectral-profile clustering — merges over-segmented pyannote speakers."""

import numpy as np

from tapeback import const
from tapeback.models import DiarizationSegment

# Minor speaker absorption: speakers with very little speech are likely
# echo/crosstalk artifacts.  Use a lower merge threshold for them.
MINOR_SPEAKER_MAX_SEC = 15.0
MINOR_SPEAKER_RATIO = 0.2
MINOR_SPEAKER_MERGE_THRESHOLD = 0.92


def _speaker_spectral_profile(
    monitor_samples: np.ndarray,
    sample_rate: int,
    diarization_segments: list[DiarizationSegment],
    speaker: str,
) -> np.ndarray:
    """Compute average power spectrum for a speaker's segments.

    Focuses on the 100-4000 Hz range which contains voice formant information.
    Uses Hann-windowed FFT with 50 % overlap.
    """
    n_fft = const.SPECTRAL_FFT_SIZE
    freq_per_bin = sample_rate / n_fft
    min_bin = max(1, int(const.SPECTRAL_MIN_FREQ_HZ / freq_per_bin))
    max_bin = min(n_fft // 2 + 1, int(const.SPECTRAL_MAX_FREQ_HZ / freq_per_bin) + 1)
    n_bins = max_bin - min_bin

    if n_bins <= 0:
        return np.zeros(1)

    window = np.hanning(n_fft)
    spectra: list[np.ndarray] = []

    for seg in diarization_segments:
        if seg.speaker != speaker:
            continue
        sf = max(0, int(seg.start * sample_rate))
        ef = min(len(monitor_samples), int(seg.end * sample_rate))
        chunk = monitor_samples[sf:ef]

        hop = n_fft // 2
        for start in range(0, len(chunk) - n_fft, hop):
            frame = chunk[start : start + n_fft].astype(np.float64)
            spectrum = np.abs(np.fft.rfft(frame * window))[min_bin:max_bin]
            spectra.append(spectrum)

    if not spectra:
        return np.zeros(n_bins)
    result: np.ndarray = np.mean(spectra, axis=0)
    return result


def _pick_merge_threshold(
    sp_a: str,
    sp_b: str,
    total_speech: dict[str, float],
    default_threshold: float,
) -> float:
    """Lower threshold when one speaker is a minor artifact (echo/crosstalk)."""
    minor_total = min(total_speech[sp_a], total_speech[sp_b])
    major_total = max(total_speech[sp_a], total_speech[sp_b])
    if (
        minor_total < MINOR_SPEAKER_MAX_SEC
        and major_total > 0
        and minor_total / major_total < MINOR_SPEAKER_RATIO
    ):
        return MINOR_SPEAKER_MERGE_THRESHOLD
    return default_threshold


def _apply_merge(merge_map: dict[str, str], sp_a: str, sp_b: str, speakers: list[str]) -> None:
    """Point every speaker currently mapped to sp_b's root at sp_a's root."""
    target = merge_map[sp_b]
    canonical = merge_map[sp_a]
    for s in speakers:
        if merge_map[s] == target:
            merge_map[s] = canonical


def merge_similar_speakers(
    diarization_segments: list[DiarizationSegment],
    monitor_samples: np.ndarray,
    sample_rate: int,
    similarity_threshold: float = 0.95,
) -> list[DiarizationSegment]:
    """Merge pyannote speakers with similar spectral profiles.

    Fixes over-segmentation where a single speaker is incorrectly split into
    multiple speakers.  Uses power-spectrum cosine similarity in the 100-4000 Hz
    voice frequency range.

    Two-tier thresholds:
    - Standard merge (similarity_threshold, default 0.96): merges near-identical
      profiles (over-segmented single speaker, cosine ~0.98-0.99).
    - Minor speaker absorption (MINOR_SPEAKER_MERGE_THRESHOLD = 0.92): when one
      speaker has very little speech (< 15s and < 20% of dominant), they are likely
      echo/crosstalk artifacts with unreliable spectral profiles. A lower threshold
      absorbs them into the dominant speaker.

    Power-spectrum similarity is a weak signal for voice identity — the channel
    frequency response dominates. Set to 0 to disable.
    """
    if similarity_threshold <= 0:
        return diarization_segments

    speakers = sorted({seg.speaker for seg in diarization_segments})
    if len(speakers) <= 1:
        return diarization_segments

    total_speech: dict[str, float] = {
        sp: sum(s.end - s.start for s in diarization_segments if s.speaker == sp) for sp in speakers
    }

    profiles: dict[str, np.ndarray] = {
        sp: _speaker_spectral_profile(monitor_samples, sample_rate, diarization_segments, sp)
        for sp in speakers
    }

    merge_map: dict[str, str] = {s: s for s in speakers}

    for i, sp_a in enumerate(speakers):
        for sp_b in speakers[i + 1 :]:
            if merge_map[sp_a] == merge_map[sp_b]:
                continue

            norm_a = float(np.linalg.norm(profiles[sp_a]))
            norm_b = float(np.linalg.norm(profiles[sp_b]))
            if norm_a < const.CHANNEL_EPSILON or norm_b < const.CHANNEL_EPSILON:
                continue

            similarity = float(np.dot(profiles[sp_a], profiles[sp_b]) / (norm_a * norm_b))
            threshold = _pick_merge_threshold(sp_a, sp_b, total_speech, similarity_threshold)

            if similarity >= threshold:
                _apply_merge(merge_map, sp_a, sp_b, speakers)

    if all(merge_map[s] == s for s in speakers):
        return diarization_segments

    return [
        DiarizationSegment(speaker=merge_map[seg.speaker], start=seg.start, end=seg.end)
        for seg in diarization_segments
    ]
