from pathlib import Path
from unittest.mock import MagicMock, call, patch

from meetrec.transcriber import Segment, Word


def test_segment_dataclass():
    """Segment and Word should create correctly."""
    word = Word(start=0.0, end=0.5, word="hello", probability=0.95)
    assert word.start == 0.0
    assert word.word == "hello"

    segment = Segment(start=0.0, end=5.0, text="hello world", words=[word])
    assert segment.text == "hello world"
    assert segment.speaker is None
    assert segment.words is not None
    assert len(segment.words) == 1


def test_transcribe_returns_segments(settings):
    """Transcriber should map faster-whisper output to Segment dataclasses."""
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 5.0
    mock_segment.text = " Hello world "

    mock_word = MagicMock()
    mock_word.start = 0.0
    mock_word.end = 0.5
    mock_word.word = "Hello"
    mock_word.probability = 0.95
    mock_segment.words = [mock_word]

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    with patch("meetrec.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([mock_segment]), mock_info)

        from meetrec.transcriber import Transcriber

        transcriber = Transcriber(settings)
        segments, info = transcriber.transcribe(Path("/fake/audio.wav"))

    assert len(segments) == 1
    assert segments[0].text == "Hello world"
    assert segments[0].words is not None
    assert len(segments[0].words) == 1
    assert info["language"] == "en"
    assert info["duration"] == 5.0


def test_cuda_fallback_to_cpu(settings):
    """Should fall back to CPU when CUDA is not available."""
    call_args = []

    def mock_init(model_name, device="cuda", compute_type="float16"):
        call_args.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA not available")
        return MagicMock()

    with patch("meetrec.transcriber.WhisperModel", side_effect=mock_init):
        from meetrec.transcriber import Transcriber

        Transcriber(settings)

    assert call_args == ["cuda", "cpu"]


def test_empty_transcription(settings, capsys):
    """Empty transcription result should return empty list with warning."""
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0

    with patch("meetrec.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), mock_info)

        from meetrec.transcriber import Transcriber

        transcriber = Transcriber(settings)
        segments, _info = transcriber.transcribe(Path("/fake/audio.wav"))

    assert segments == []
    captured = capsys.readouterr()
    assert "No speech detected" in captured.err


def test_cuda_inference_fallback_to_cpu(settings, capsys):
    """Model loads on CUDA but fails during inference (e.g. missing libcublas).

    Should recreate the model on CPU and retry transcription.
    """
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 5.0
    mock_segment.text = "Hello"
    mock_segment.words = []

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    def failing_iter():
        """Generator that raises RuntimeError on first iteration (simulates CUDA failure)."""
        raise RuntimeError("Library libcublas.so.12 is not found")
        yield  # make it a generator

    with patch("meetrec.transcriber.WhisperModel") as mock_model_cls:
        cuda_model = MagicMock()
        cpu_model = MagicMock()
        # First WhisperModel() returns cuda_model, second returns cpu_model
        mock_model_cls.side_effect = [cuda_model, cpu_model]

        # CUDA model: transcribe returns an iterator that raises during iteration
        cuda_model.transcribe.return_value = (failing_iter(), mock_info)
        # CPU model: transcribe returns normal segments
        cpu_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        from meetrec.transcriber import Transcriber

        transcriber = Transcriber(settings)
        segments, info = transcriber.transcribe(Path("/fake/audio.wav"))

    # Verify fallback happened: two WhisperModel calls (cuda, then cpu)
    assert mock_model_cls.call_count == 2
    assert mock_model_cls.call_args_list[0] == call(
        settings.whisper_model, device="cuda", compute_type="float16"
    )
    assert mock_model_cls.call_args_list[1] == call(
        settings.whisper_model, device="cpu", compute_type="int8"
    )

    # Verify transcription succeeded on CPU
    assert len(segments) == 1
    assert segments[0].text == "Hello"
    assert info["duration"] == 5.0

    captured = capsys.readouterr()
    assert "CUDA runtime error" in captured.err
