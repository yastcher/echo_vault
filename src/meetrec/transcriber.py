import sys
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from meetrec.settings import Settings


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    start: float  # seconds
    end: float  # seconds
    text: str
    words: list[Word] | None = None
    speaker: str | None = None  # None in phase 1, filled in phase 2


class Transcriber:
    def __init__(self, settings: Settings) -> None:
        """Initialize faster-whisper model.

        Falls back from CUDA to CPU if CUDA is not available.
        First run downloads the model automatically.
        """
        self._settings = settings
        device = settings.device
        compute_type = settings.compute_type

        try:
            self._model = WhisperModel(
                settings.whisper_model,
                device=device,
                compute_type=compute_type,
            )
        except RuntimeError:
            if device == "cuda":
                print(
                    "Warning: CUDA not available, falling back to CPU",
                    file=sys.stderr,
                )
                self._model = WhisperModel(
                    settings.whisper_model,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise

    def transcribe(self, audio_path: Path) -> tuple[list[Segment], dict]:
        """Transcribe audio file.

        Returns (list of Segments, info dict with language/duration/etc).
        """
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=self._settings.language,
            beam_size=self._settings.beam_size,
            vad_filter=self._settings.vad_filter,
            word_timestamps=True,
        )

        segments: list[Segment] = []
        for seg in segments_iter:
            words: list[Word] | None = None
            if seg.words:
                words = [
                    Word(
                        start=w.start,
                        end=w.end,
                        word=w.word,
                        probability=w.probability,
                    )
                    for w in seg.words
                ]

            segments.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=words,
                )
            )

        if not segments:
            print("Warning: No speech detected in audio", file=sys.stderr)

        info_dict = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

        return segments, info_dict
