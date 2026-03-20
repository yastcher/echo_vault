from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_transcribe_stereo(settings):
    """transcribe_stereo should call transcribe twice and assign speakers correctly."""
    mock_seg_mic = MagicMock()
    mock_seg_mic.start = 0.0
    mock_seg_mic.end = 3.0
    mock_seg_mic.text = " My speech "
    mock_seg_mic.words = []

    mock_seg_monitor = MagicMock()
    mock_seg_monitor.start = 1.0
    mock_seg_monitor.end = 4.0
    mock_seg_monitor.text = " Their speech "
    mock_seg_monitor.words = []

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    with patch("meetrec.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        # First call: mic, second call: monitor
        instance.transcribe.side_effect = [
            (iter([mock_seg_mic]), mock_info),
            (iter([mock_seg_monitor]), mock_info),
        ]

        from meetrec.transcriber import Transcriber

        transcriber = Transcriber(settings)
        mic_segs, monitor_segs, _info = transcriber.transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav")
        )

    # Whisper called twice
    assert instance.transcribe.call_count == 2

    # Mic segments get speaker="You"
    assert len(mic_segs) == 1
    assert mic_segs[0].speaker == "You"
    assert mic_segs[0].text == "My speech"

    # Monitor segments keep speaker=None
    assert len(monitor_segs) == 1
    assert monitor_segs[0].speaker is None
    assert monitor_segs[0].text == "Their speech"


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
