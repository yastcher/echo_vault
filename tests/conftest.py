import pytest

from meetrec.settings import Settings

pytest_plugins = [
    "tests.fixtures",
]


@pytest.fixture
def tmp_vault(tmp_path):
    """Temporary Obsidian vault for tests."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def settings(tmp_vault):
    """Settings with temporary vault."""
    return Settings(vault_path=tmp_vault)
