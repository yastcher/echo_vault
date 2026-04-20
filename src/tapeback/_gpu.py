"""GPU memory helpers — shared between pipeline and live transcription."""

import gc


def free_gpu_memory() -> None:
    """Release Python refs and clear CUDA cache so the next model fits in VRAM."""
    gc.collect()
    try:
        import torch  # noqa: PLC0415 — optional dependency, guarded by try/except

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
