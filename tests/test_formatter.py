from meetrec.formatter import format_markdown, save_to_vault
from meetrec.transcriber import Segment


def test_format_markdown_has_frontmatter():
    """Markdown should have YAML frontmatter with date, duration, audio wikilink."""
    segments = [
        Segment(start=83.0, end=90.0, text="Lorem ipsum dolor sit amet."),
    ]

    result = format_markdown(
        segments=segments,
        session_name="2026-03-17_14-30-00",
        audio_rel_path="attachments/audio/2026-03-17_14-30-00.wav",
        duration_seconds=5025.0,
        language="en",
    )

    assert result.startswith("---\n")
    assert "date: 2026-03-17" in result
    assert 'time: "14:30"' in result
    assert 'duration: "01:23:45"' in result
    assert "language: en" in result
    assert "[[attachments/audio/2026-03-17_14-30-00.wav]]" in result
    assert "  - meeting" in result
    assert "  - transcript" in result


def test_format_markdown_has_timecodes():
    """Each segment should have [HH:MM:SS] timecode."""
    segments = [
        Segment(start=83.0, end=90.0, text="First segment.", speaker="Speaker 1"),
        Segment(start=165.0, end=170.0, text="Second segment.", speaker="Speaker 2"),
    ]

    result = format_markdown(
        segments=segments,
        session_name="2026-03-17_14-30-00",
        audio_rel_path="audio.wav",
        duration_seconds=300.0,
        language="en",
    )

    assert "[00:01:23]" in result
    assert "First segment." in result
    assert "[00:02:45]" in result
    assert "Second segment." in result


def test_short_segments_filtered():
    """Segments shorter than 1 second should be filtered out."""
    segments = [
        Segment(start=0.0, end=0.5, text="Too short."),
        Segment(start=1.0, end=5.0, text="Long enough."),
        Segment(start=10.0, end=10.8, text="Also too short."),
    ]

    result = format_markdown(
        segments=segments,
        session_name="2026-03-17_14-30-00",
        audio_rel_path="audio.wav",
        duration_seconds=60.0,
        language="en",
    )

    assert "Too short" not in result
    assert "Also too short" not in result
    assert "Long enough" in result


def test_consecutive_speakers_merged():
    """Consecutive segments from the same speaker should be merged."""
    segments = [
        Segment(start=0.0, end=5.0, text="Hello there.", speaker="You"),
        Segment(start=5.0, end=10.0, text="How are you?", speaker="You"),
        Segment(start=10.0, end=15.0, text="I'm fine.", speaker="Speaker 1"),
        Segment(start=15.0, end=20.0, text="Thanks.", speaker="Speaker 1"),
        Segment(start=20.0, end=25.0, text="Great.", speaker="You"),
    ]

    result = format_markdown(
        segments=segments,
        session_name="2026-03-17_14-30-00",
        audio_rel_path="audio.wav",
        duration_seconds=25.0,
        language="en",
    )

    # Two "You" segments merged into one line
    assert "**You:** Hello there. How are you?" in result
    # Two "Speaker 1" segments merged
    assert "**Speaker 1:** I'm fine. Thanks." in result
    # Third "You" block is separate
    assert "**You:** Great." in result
    # Only 3 timecoded lines, not 5
    assert result.count("[00:") == 3


def test_save_to_vault_creates_files(settings, tmp_vault, tmp_path):
    """save_to_vault should create .md and .wav in correct subdirectories."""
    stereo_wav = tmp_path / "stereo.wav"
    stereo_wav.write_bytes(b"fake wav data")

    md_path = save_to_vault(
        markdown="# Test",
        stereo_wav=stereo_wav,
        settings=settings,
        session_name="2026-03-17_14-30-00",
    )

    assert md_path.exists()
    assert md_path.read_text() == "# Test"
    assert md_path.name == "2026-03-17_14-30-00.md"

    audio_path = tmp_vault / "attachments" / "audio" / "2026-03-17_14-30-00.wav"
    assert audio_path.exists()


def test_save_no_overwrite(settings, tmp_vault, tmp_path):
    """Existing files should get _1 suffix instead of being overwritten."""
    stereo_wav = tmp_path / "stereo.wav"
    stereo_wav.write_bytes(b"fake wav data")

    # Create first version
    save_to_vault("# First", stereo_wav, settings, "test")

    # Create second version — should get _1 suffix
    md_path = save_to_vault("# Second", stereo_wav, settings, "test")

    assert md_path.name == "test_1.md"
    assert md_path.read_text() == "# Second"


def test_save_creates_directories(settings, tmp_vault, tmp_path):
    """meetings_dir and attachments_dir should be created if missing."""
    stereo_wav = tmp_path / "stereo.wav"
    stereo_wav.write_bytes(b"fake wav data")

    meetings_dir = tmp_vault / settings.meetings_dir
    attachments_dir = tmp_vault / settings.attachments_dir

    assert not meetings_dir.exists()
    assert not attachments_dir.exists()

    save_to_vault("# Test", stereo_wav, settings, "test")

    assert meetings_dir.exists()
    assert attachments_dir.exists()
