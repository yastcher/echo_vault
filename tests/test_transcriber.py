from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meetrec.transcriber import Segment, Transcriber, Word


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


# --- split_on_pauses ---


def test_split_on_pauses_splits_at_gap():
    """Long segment with a pause between words should be split into two."""
    seg = Segment(
        start=0.0,
        end=10.0,
        text="I talk alone we talk together",
        words=[
            Word(start=0.0, end=1.0, word="I", probability=0.9),
            Word(start=1.0, end=2.0, word="talk", probability=0.9),
            Word(start=2.0, end=3.0, word="alone", probability=0.9),
            # 2-second pause here (3.0 → 5.0)
            Word(start=5.0, end=6.0, word="we", probability=0.9),
            Word(start=6.0, end=7.0, word="talk", probability=0.9),
            Word(start=7.0, end=8.0, word="together", probability=0.9),
        ],
        speaker="You",
    )

    result = Transcriber.split_on_pauses([seg], pause_threshold=1.0)

    assert len(result) == 2
    assert result[0].text == "I talk alone"
    assert result[0].start == 0.0
    assert result[0].end == 3.0
    assert result[0].speaker == "You"
    assert result[1].text == "we talk together"
    assert result[1].start == 5.0
    assert result[1].end == 8.0
    assert result[1].speaker == "You"


def test_split_on_pauses_no_split_when_continuous():
    """Segment without pauses should pass through unchanged."""
    seg = Segment(
        start=0.0,
        end=3.0,
        text="hello world today",
        words=[
            Word(start=0.0, end=1.0, word="hello", probability=0.9),
            Word(start=1.0, end=2.0, word="world", probability=0.9),
            Word(start=2.0, end=3.0, word="today", probability=0.9),
        ],
    )

    result = Transcriber.split_on_pauses([seg], pause_threshold=1.0)

    assert len(result) == 1
    assert result[0].text == "hello world today"


def test_split_on_pauses_no_words():
    """Segment without words should pass through unchanged."""
    seg = Segment(start=0.0, end=5.0, text="no words here")

    result = Transcriber.split_on_pauses([seg], pause_threshold=1.0)

    assert len(result) == 1
    assert result[0] is seg


@pytest.mark.parametrize(
    "threshold,expected_count",
    [
        # Gaps: a→b=0.8s, b→c=2.0s, c→d=4.0s
        pytest.param(0.5, 4, id="tight_threshold"),  # all 3 gaps >= 0.5 → 4 chunks
        pytest.param(1.5, 3, id="medium_threshold"),  # 2 gaps >= 1.5 (2.0, 4.0) → 3 chunks
        pytest.param(5.0, 1, id="loose_threshold"),  # no gaps >= 5.0 → 1 chunk
    ],
)
def test_split_on_pauses_threshold_sensitivity(threshold, expected_count):
    """Different thresholds should produce different split counts."""
    seg = Segment(
        start=0.0,
        end=20.0,
        text="a b c d",
        words=[
            Word(start=0.0, end=1.0, word="a", probability=0.9),
            # 0.8s gap
            Word(start=1.8, end=2.5, word="b", probability=0.9),
            # 2.0s gap
            Word(start=4.5, end=5.5, word="c", probability=0.9),
            # 4.0s gap
            Word(start=9.5, end=10.0, word="d", probability=0.9),
        ],
    )

    result = Transcriber.split_on_pauses([seg], pause_threshold=threshold)

    assert len(result) == expected_count
