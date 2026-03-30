"""System tray tests — state transitions, icon generation, menu callbacks."""

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.fixtures import requires_pystray

pytestmark = requires_pystray

with contextlib.suppress(Exception):
    # Fails on headless CI (no X display); all tests skipped via requires_pystray
    from tapeback.tray import TrayApp, TrayState, _create_icon, _icon_for_state


# --- Icon generation ---


def test_create_icon_returns_rgba_image():
    """Generated icon should be 64x64 RGBA PIL Image."""
    icon = _create_icon("#FF0000")
    assert icon.size == (64, 64)
    assert icon.mode == "RGBA"


def test_icon_for_each_state():
    """Each state produces a distinct icon image."""
    icons = {state: _icon_for_state(state) for state in TrayState}
    assert len(icons) == 3
    for icon in icons.values():
        assert icon.size == (64, 64)


# --- Initial state ---


def test_initial_state_idle(tray_app):
    """TrayApp starts in IDLE state."""
    assert tray_app._state == TrayState.IDLE


def test_initial_state_picks_up_existing_recording(settings):
    """TrayApp detects an in-progress recording at startup."""
    app = TrayApp(settings)
    app._recorder = MagicMock()
    app._recorder.is_recording.return_value = True

    # run() is blocking (pystray event loop), so mock Icon to capture setup
    with patch("tapeback.tray.pystray.Icon") as mock_icon_cls:
        mock_icon_cls.return_value.run = MagicMock()
        app.run()

    assert app._state == TrayState.RECORDING


# --- Start recording ---


def test_on_start_spawns_thread_when_idle(tray_app):
    """_on_start spawns a thread only when state is IDLE."""
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_start(MagicMock(), MagicMock())
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    assert tray_app._state == TrayState.RECORDING


def test_on_start_ignored_when_recording(tray_app):
    """_on_start is a no-op when already recording."""
    tray_app._state = TrayState.RECORDING
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_start(MagicMock(), MagicMock())
        mock_thread.assert_not_called()


def test_on_start_ignored_when_processing(tray_app):
    """_on_start is a no-op during processing."""
    tray_app._state = TrayState.PROCESSING
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_start(MagicMock(), MagicMock())
        mock_thread.assert_not_called()


def test_do_start_success(tray_app):
    """Successful _do_start keeps RECORDING state and notifies."""
    tray_app._state = TrayState.RECORDING
    tray_app._recorder.start.return_value = "2026-03-30_10-00-00"

    tray_app._do_start()

    tray_app._recorder.start.assert_called_once_with(tray_app._settings)
    assert tray_app._state == TrayState.RECORDING


def test_do_start_failure_resets_to_idle(tray_app):
    """Failed _do_start resets state to IDLE."""
    tray_app._state = TrayState.RECORDING
    tray_app._recorder.start.side_effect = RuntimeError("parecord not found")

    tray_app._do_start()

    assert tray_app._state == TrayState.IDLE


# --- Stop recording ---


def test_on_stop_spawns_thread_when_recording(tray_app):
    """_on_stop spawns a thread only when state is RECORDING."""
    tray_app._state = TrayState.RECORDING
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_stop(MagicMock(), MagicMock())
        mock_thread.assert_called_once()

    assert tray_app._state == TrayState.PROCESSING


def test_on_stop_ignored_when_idle(tray_app):
    """_on_stop is a no-op when idle."""
    with patch("tapeback.tray.threading.Thread") as mock_thread:
        tray_app._on_stop(MagicMock(), MagicMock())
        mock_thread.assert_not_called()


def test_do_stop_and_process_success(tray_app):
    """Successful processing resets to IDLE and notifies."""
    tray_app._state = TrayState.PROCESSING

    with patch("tapeback.tray.stop_and_process", return_value=Path("/vault/meetings/test.md")):
        tray_app._do_stop_and_process()

    assert tray_app._state == TrayState.IDLE
    tray_app._icon.notify.assert_called()


def test_do_stop_and_process_failure_resets_to_idle(tray_app):
    """Failed processing resets to IDLE."""
    tray_app._state = TrayState.PROCESSING

    with patch("tapeback.tray.stop_and_process", side_effect=RuntimeError("transcription failed")):
        tray_app._do_stop_and_process()

    assert tray_app._state == TrayState.IDLE


# --- Status ---


def test_on_status_while_recording(tray_app):
    """Status shows session info while recording."""
    tray_app._state = TrayState.RECORDING
    tray_app._recorder.get_session_info.return_value = {
        "session_name": "test-meeting",
        "started_at": "2026-03-30T10:00:00",
    }

    tray_app._on_status(MagicMock(), MagicMock())

    tray_app._icon.notify.assert_called_once()
    msg = tray_app._icon.notify.call_args[0][0]
    assert "test-meeting" in msg


def test_on_status_while_idle(tray_app):
    """Status shows idle message when not recording."""
    tray_app._on_status(MagicMock(), MagicMock())

    tray_app._icon.notify.assert_called_once()
    msg = tray_app._icon.notify.call_args[0][0]
    assert "Idle" in msg


# --- Quit ---


def test_quit_during_recording_stops_recorder(tray_app):
    """Quit while recording stops the recorder and preserves audio."""
    tray_app._state = TrayState.RECORDING

    tray_app._on_quit(MagicMock(), MagicMock())

    tray_app._recorder.stop.assert_called_once()
    tray_app._icon.stop.assert_called_once()


def test_quit_while_idle(tray_app):
    """Quit while idle just stops the icon."""
    tray_app._on_quit(MagicMock(), MagicMock())

    tray_app._recorder.stop.assert_not_called()
    tray_app._icon.stop.assert_called_once()


def test_quit_during_processing(tray_app):
    """Quit during processing stops icon without stopping recorder."""
    tray_app._state = TrayState.PROCESSING

    tray_app._on_quit(MagicMock(), MagicMock())

    tray_app._recorder.stop.assert_not_called()
    tray_app._icon.stop.assert_called_once()
