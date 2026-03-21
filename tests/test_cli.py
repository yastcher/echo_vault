"""CLI integration tests — invoke click commands via CliRunner.

Mock only ML models (WhisperModel, pyannote Pipeline). Everything else
(ffmpeg, file I/O, formatting) runs for real.
"""

import shutil
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from meetrec.cli import cli
from tests.fixtures import create_mono_wav, create_silent_wav, create_stereo_wav_segments


@pytest.fixture
def runner():
    return CliRunner()


def _mock_whisper_transcribe(segments_data):
    """Create a mock WhisperModel that returns given segments."""
    mock_model = MagicMock()

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0

    mock_segments = []
    for start, end, text in segments_data:
        seg = MagicMock()
        seg.start = start
        seg.end = end
        seg.text = f" {text} "
        seg.words = []
        mock_segments.append(seg)

    # Each call to transcribe() returns a fresh iterator
    mock_model.transcribe.side_effect = lambda *a, **kw: (iter(list(mock_segments)), mock_info)

    return mock_model


# --- process command ---


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_mono_file(runner, tmp_path, monkeypatch):
    """process command: mono WAV → transcribe → save markdown + audio to vault."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEETREC_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEETREC_DIARIZE", "false")

    # Create input audio
    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_silent_wav(audio, duration=2.0, sample_rate=48000)

    mock_model = _mock_whisper_transcribe(
        [
            (0.0, 5.0, "Hello from the meeting."),
            (5.0, 10.0, "Second sentence here."),
        ]
    )

    with patch("meetrec.transcriber.WhisperModel", return_value=mock_model):
        result = runner.invoke(cli, ["process", str(audio), "--no-diarize"])

    err_msg = result.output + (str(result.exception) if result.exception else "")
    assert result.exit_code == 0, err_msg

    # Markdown saved to vault
    md_path = vault / "meetings" / "2026-03-20_10-00-00.md"
    assert md_path.exists()
    md_content = md_path.read_text()
    assert "Hello from the meeting." in md_content
    assert "Second sentence here." in md_content
    assert "date: 2026-03-20" in md_content

    # Audio copied to vault
    audio_dest = vault / "attachments" / "audio" / "2026-03-20_10-00-00.wav"
    assert audio_dest.exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_with_custom_name(runner, tmp_path, monkeypatch):
    """process --name uses custom name for output files."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEETREC_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEETREC_DIARIZE", "false")

    audio = tmp_path / "recording.wav"
    create_silent_wav(audio, duration=2.0)

    mock_model = _mock_whisper_transcribe([(0.0, 5.0, "Speech.")])

    with patch("meetrec.transcriber.WhisperModel", return_value=mock_model):
        result = runner.invoke(cli, ["process", str(audio), "--name", "my-meeting", "--no-diarize"])

    assert result.exit_code == 0, result.output
    assert (vault / "meetings" / "my-meeting.md").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_with_diarization(runner, tmp_path, monkeypatch):
    """process command with diarization: transcribe → diarize → speakers in markdown."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEETREC_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEETREC_HF_TOKEN", "hf_fake")

    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_silent_wav(audio, duration=2.0)

    mock_model = _mock_whisper_transcribe(
        [
            (0.0, 5.0, "First speaker."),
            (5.0, 10.0, "Second speaker."),
        ]
    )

    mock_turn_1 = MagicMock()
    mock_turn_1.start = 0.0
    mock_turn_1.end = 5.0
    mock_turn_2 = MagicMock()
    mock_turn_2.start = 5.0
    mock_turn_2.end = 10.0

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (mock_turn_1, None, "SPEAKER_00"),
        (mock_turn_2, None, "SPEAKER_01"),
    ]

    with (
        patch("meetrec.transcriber.WhisperModel", return_value=mock_model),
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = mock_annotation

        result = runner.invoke(cli, ["process", str(audio)])

    assert result.exit_code == 0, result.output

    md_content = (vault / "meetings" / "2026-03-20_10-00-00.md").read_text()
    assert "Speaker 1" in md_content
    assert "Speaker 2" in md_content


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_file(runner, tmp_path, monkeypatch):
    """process command with stereo WAV: detects stereo, uses for channel attribution."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEETREC_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEETREC_HF_TOKEN", "hf_fake")

    audio = tmp_path / "2026-03-20_10-00-00.wav"
    create_stereo_wav_segments(
        audio,
        sample_rate=48000,
        segments_spec=[
            (1.0, 0.8, 0.003),  # mic dominant
            (1.0, 0.003, 0.8),  # monitor dominant
        ],
    )

    mock_model = _mock_whisper_transcribe(
        [
            (0.0, 5.0, "User speech."),
            (5.0, 10.0, "Remote speech."),
        ]
    )

    mock_turn_1 = MagicMock()
    mock_turn_1.start = 0.0
    mock_turn_1.end = 5.0
    mock_turn_2 = MagicMock()
    mock_turn_2.start = 5.0
    mock_turn_2.end = 10.0

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [
        (mock_turn_1, None, "SPEAKER_00"),
        (mock_turn_2, None, "SPEAKER_01"),
    ]

    with (
        patch("meetrec.transcriber.WhisperModel", return_value=mock_model),
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = mock_annotation

        result = runner.invoke(cli, ["process", str(audio)])

    assert result.exit_code == 0, result.output

    md_content = (vault / "meetings" / "2026-03-20_10-00-00.md").read_text()
    # Stereo attribution should identify user speaker
    assert "User speech." in md_content
    assert "Remote speech." in md_content


# --- status command ---


def test_status_not_recording(runner, tmp_path, monkeypatch):
    """status command when not recording."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEETREC_VAULT_PATH", str(vault))

    with patch("meetrec.cli.get_settings") as mock_settings:
        from meetrec.settings import Settings

        mock_settings.return_value = Settings(vault_path=vault)

        with patch("meetrec.recorder.Recorder.get_session_info", return_value=None):
            result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0
    assert "Not recording." in result.output
    assert str(vault) in result.output


def test_status_while_recording(runner, tmp_path, monkeypatch):
    """status command when recording is in progress."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEETREC_VAULT_PATH", str(vault))

    session_info = {
        "session_name": "2026-03-20_10-00-00",
        "started_at": "2026-03-20T10:00:00",
        "pid_monitor": 12345,
        "pid_mic": 12346,
        "monitor_path": "/tmp/meetrec/test/monitor.wav",
        "mic_path": "/tmp/meetrec/test/mic.wav",
    }

    with patch("meetrec.cli.get_settings") as mock_settings:
        from meetrec.settings import Settings

        mock_settings.return_value = Settings(vault_path=vault)

        with patch("meetrec.recorder.Recorder.get_session_info", return_value=session_info):
            result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0
    assert "Recording in progress: 2026-03-20_10-00-00" in result.output


# --- _stop_and_process (dual-channel pipeline) ---


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_stop_and_process_pipeline(tmp_path):
    """_stop_and_process: full dual-channel pipeline with mocked ML models."""
    from meetrec.cli import _stop_and_process
    from meetrec.settings import Settings

    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault, hf_token="hf_fake")

    # Create monitor.wav and mic.wav in a session directory
    session_dir = tmp_path / "2026-03-20_10-00-00"
    session_dir.mkdir()
    monitor_wav = session_dir / "monitor.wav"
    mic_wav = session_dir / "mic.wav"
    create_mono_wav(monitor_wav, duration=2.0, sample_rate=48000, amplitude=0.5)
    create_mono_wav(mic_wav, duration=2.0, sample_rate=48000, amplitude=0.5)

    # Mock recorder.stop() to return our files
    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = (monitor_wav, mic_wav)

    # Mock Whisper — segment times must fit within audio duration (2s)
    mock_model = _mock_whisper_transcribe(
        [
            (0.0, 1.0, "Hello."),
            (1.0, 2.0, "World."),
        ]
    )

    # Mock pyannote
    mock_turn = MagicMock()
    mock_turn.start = 0.0
    mock_turn.end = 2.0
    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = [(mock_turn, None, "SPEAKER_00")]

    with (
        patch("meetrec.transcriber.WhisperModel", return_value=mock_model),
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_pipeline.return_value = mock_annotation

        _stop_and_process(mock_recorder, settings, diarize=True)

    # Verify outputs
    md_path = vault / "meetings" / "2026-03-20_10-00-00.md"
    assert md_path.exists()
    md_content = md_path.read_text()
    assert "Hello." in md_content
    assert "World." in md_content

    audio_dest = vault / "attachments" / "audio" / "2026-03-20_10-00-00.wav"
    assert audio_dest.exists()

    # Temp files cleaned up
    assert not session_dir.exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_stop_and_process_no_diarize(tmp_path):
    """_stop_and_process with diarize=False skips pyannote entirely."""
    from meetrec.cli import _stop_and_process
    from meetrec.settings import Settings

    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    session_dir = tmp_path / "2026-03-20_11-00-00"
    session_dir.mkdir()
    monitor_wav = session_dir / "monitor.wav"
    mic_wav = session_dir / "mic.wav"
    create_mono_wav(monitor_wav, duration=2.0, sample_rate=48000, amplitude=0.5)
    create_mono_wav(mic_wav, duration=2.0, sample_rate=48000, amplitude=0.5)

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = (monitor_wav, mic_wav)

    mock_model = _mock_whisper_transcribe([(0.0, 5.0, "No diarize.")])

    with (
        patch("meetrec.transcriber.WhisperModel", return_value=mock_model),
        patch("pyannote.audio.Pipeline") as mock_pipeline_cls,
    ):
        _stop_and_process(mock_recorder, settings, diarize=False)

        # pyannote never called
        mock_pipeline_cls.from_pretrained.assert_not_called()

    md_path = vault / "meetings" / "2026-03-20_11-00-00.md"
    assert md_path.exists()


# --- _maybe_diarize ---


def test_maybe_diarize_skips_when_disabled(tmp_path):
    """_maybe_diarize returns segments unchanged when diarize=False."""
    from meetrec.cli import _maybe_diarize
    from meetrec.settings import Settings
    from meetrec.transcriber import Segment

    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    segments = [Segment(start=0.0, end=5.0, text="Hello")]
    result = _maybe_diarize(segments, settings, tmp_path / "audio.wav", None, diarize=False)
    assert result is segments


def test_maybe_diarize_warns_without_token(runner, tmp_path, capsys):
    """_maybe_diarize warns and skips when HF token is not set."""
    from meetrec.cli import _maybe_diarize
    from meetrec.settings import Settings
    from meetrec.transcriber import Segment

    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault, hf_token="", diarize=True)

    segments = [Segment(start=0.0, end=5.0, text="Hello")]
    result = _maybe_diarize(segments, settings, tmp_path / "audio.wav", None, diarize=True)
    assert result is segments
