"""Regression tests for path-traversal vulnerabilities.

Bug: `tapeback process --name "../../etc/passwd"` bypassed the session-name
regex check (which only `recorder.start` enforced) and let `save_audio_to_vault`
/ `save_markdown_to_vault` write files outside the vault root.
"""

import pytest

from tapeback.pipeline import process_file
from tapeback.settings import Settings
from tapeback.vault import save_audio_to_vault, save_markdown_to_vault
from tests.fixtures import create_mono_wav


@pytest.mark.parametrize(
    "malicious_name",
    [
        "../../tmp/escape",
        "../escape",
        "sub/dir/escape",
        "name with spaces",
        "name;rm -rf /",
    ],
)
def test_process_file_rejects_bad_session_name(tmp_path, malicious_name):
    """process_file must reject names with path separators or unsafe chars."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    audio = tmp_path / "input.wav"
    create_mono_wav(audio, duration=0.5)

    with pytest.raises(ValueError, match="session name"):
        process_file(audio, settings, name=malicious_name, diarize=False, do_summarize=False)


def test_save_audio_to_vault_rejects_traversal(tmp_path):
    """save_audio_to_vault must not write outside vault_path."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    audio = tmp_path / "input.wav"
    create_mono_wav(audio, duration=0.5)

    with pytest.raises(ValueError, match="session name"):
        save_audio_to_vault(audio, settings, "../escape")


def test_save_markdown_to_vault_rejects_traversal(tmp_path):
    """save_markdown_to_vault must not write outside vault_path."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    with pytest.raises(ValueError, match="session name"):
        save_markdown_to_vault("content", settings, "../escape")
