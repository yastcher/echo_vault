import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import click

from meetrec.settings import get_settings


@click.group()
def cli():
    """meetrec — local meeting recorder for Obsidian."""


@cli.command()
@click.argument("name", required=False)
def start(name):
    """Start recording monitor source + microphone.

    Runs in foreground — press Ctrl+C to stop and transcribe.
    """
    settings = get_settings()

    from meetrec.recorder import Recorder, detect_devices

    recorder = Recorder()

    monitor, mic = detect_devices(settings)
    session_name = recorder.start(settings, session_name=name)

    click.echo(f"Recording started: {session_name}", err=True)
    click.echo(f"Monitor: {monitor}", err=True)
    click.echo(f"Mic: {mic}", err=True)
    click.echo("Run 'meetrec stop' to finish and transcribe.", err=True)
    click.echo("Or press Ctrl+C to stop and transcribe now.", err=True)

    # Block and wait for Ctrl+C
    try:
        signal.pause()
    except KeyboardInterrupt:
        click.echo("\nStopping...", err=True)
        try:
            _stop_and_process(recorder, settings)
        except KeyboardInterrupt:
            click.echo("\nAborted during processing. Audio files kept in /tmp/meetrec/", err=True)


@cli.command()
def stop():
    """Stop recording, transcribe, and save to vault."""
    settings = get_settings()

    from meetrec.recorder import Recorder

    recorder = Recorder()
    _stop_and_process(recorder, settings)


def _stop_and_process(recorder, settings):
    """Stop recording and run the full processing pipeline."""
    click.echo("Stopping recording...", err=True)
    monitor_path, mic_path = recorder.stop()

    from meetrec.audio import merge_channels
    from meetrec.formatter import format_markdown, save_to_vault
    from meetrec.transcriber import Transcriber

    click.echo("Merging audio channels...", err=True)
    output_dir = monitor_path.parent
    stereo_path, mono_16k_path = merge_channels(monitor_path, mic_path, output_dir)

    click.echo("Transcribing (this may take a few minutes)...", err=True)
    transcriber = Transcriber(settings)
    segments, info = transcriber.transcribe(mono_16k_path)

    session_name = monitor_path.parent.name
    audio_rel_path = f"{settings.attachments_dir}/{session_name}.wav"

    markdown = format_markdown(
        segments=segments,
        session_name=session_name,
        audio_rel_path=audio_rel_path,
        duration_seconds=info.get("duration", 0.0),
        language=info.get("language", settings.language),
    )

    md_path = save_to_vault(markdown, stereo_path, settings, session_name)

    # Clean up temp files
    shutil.rmtree(monitor_path.parent, ignore_errors=True)

    click.echo(f"Saved: {md_path}", err=True)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--name", default=None, help="Session name for output file")
def process(audio_file, name):
    """Process an existing audio file (mp3, m4a, ogg, wav)."""
    settings = get_settings()

    from meetrec.audio import convert_to_mono16k
    from meetrec.formatter import format_markdown, save_to_vault
    from meetrec.transcriber import Transcriber

    audio_path = Path(audio_file)

    if name is None:
        name = audio_path.stem

    click.echo("Converting audio...", err=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="meetrec_"))
    mono_16k_path = convert_to_mono16k(audio_path, tmp_dir)

    click.echo("Transcribing (this may take a few minutes)...", err=True)
    transcriber = Transcriber(settings)
    segments, info = transcriber.transcribe(mono_16k_path)

    audio_rel_path = f"{settings.attachments_dir}/{name}.wav"

    markdown = format_markdown(
        segments=segments,
        session_name=name,
        audio_rel_path=audio_rel_path,
        duration_seconds=info.get("duration", 0.0),
        language=info.get("language", settings.language),
    )

    md_path = save_to_vault(markdown, audio_path, settings, name)

    # Clean up temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    click.echo(f"Saved: {md_path}", err=True)


@cli.command()
def status():
    """Show current recording status and settings."""
    settings = get_settings()

    from meetrec.recorder import Recorder

    recorder = Recorder()
    session = recorder.get_session_info()

    if session:
        click.echo(f"Recording in progress: {session['session_name']}")
        click.echo(f"Started at: {session['started_at']}")
    else:
        click.echo("Not recording.")

    click.echo(f"\nVault: {settings.vault_path}")
    click.echo(f"Whisper model: {settings.whisper_model}")
    click.echo(f"Device: {settings.device}")
    click.echo(f"Language: {settings.language}")

    # Show available audio devices
    if shutil.which("pactl"):
        click.echo("\nAudio sources:")
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            click.echo(result.stdout)
