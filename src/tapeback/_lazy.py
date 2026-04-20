"""Lazy loaders for heavy ML dependencies.

Loading `tapeback.transcriber` drags in `faster_whisper` and `torch`
(~10 seconds cold-start). Keep the import inside the call so that
`tapeback --help` and `tapeback status` stay fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tapeback.settings import Settings
    from tapeback.transcriber import Transcriber


def load_transcriber(settings: Settings) -> Transcriber:
    """Import and instantiate Transcriber on demand."""
    from tapeback.transcriber import Transcriber  # noqa: PLC0415 — 10s ML import, must stay lazy

    return Transcriber(settings)
