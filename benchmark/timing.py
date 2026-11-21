"""Shared Real-Time Factor (RTF) timing helper for ASR models."""

import time


class RTFTimer:
    """Measure processing time and compute Real-Time Factor (RTF = proc_s / audio_s)."""

    def __init__(self) -> None:
        self._start: float | None = None
        self.elapsed_s: float = 0.0

    def __enter__(self) -> "RTFTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None:
            self.elapsed_s += time.perf_counter() - self._start
            self._start = None

    def rtf(self, total_audio_s: float) -> float:
        """Return Real-Time Factor given total audio duration in seconds."""
        if total_audio_s <= 0:
            raise ValueError("total_audio_s must be positive")
        return self.elapsed_s / total_audio_s