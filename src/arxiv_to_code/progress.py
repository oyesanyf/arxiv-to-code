"""
Real-time CLI Progress Bar (`tqdm`) and Animated Spinner utilities for ArXiv-to-Code.

Guarantees that the user is NEVER left staring at a silent console during:
- ModelFusion DynaRot hardware & Ollama probes
- Live arXiv search & Google LangExtract 3-class grounding
- Paper-to-Python code synthesis
- ShinkaEvolve structural discovery generations
- OpenEvolve multi-island MAP-Elites micro-mutations & parallel cross-pollination
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:
    from tqdm import tqdm as _RealTqdm  # type: ignore
except ImportError:
    _RealTqdm = None  # type: ignore


class ConsoleSpinner:
    """
    Lightweight daemon-thread spinner that animates live status and elapsed time
    in the terminal during blocking I/O or compute steps, then clears cleanly on exit.
    """

    FRAMES = ("|", "/", "-", "\\")

    def __init__(self, label: str, enabled: bool = True) -> None:
        self.label = label
        self.enabled = enabled
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ts = 0.0

    def __enter__(self) -> "ConsoleSpinner":
        self._start_ts = time.perf_counter()
        if not self.enabled:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._thread.start()
        return self

    def _spin_loop(self) -> None:
        is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        if not is_tty:
            sys.stderr.write(f"[...] {self.label}...\n")
            sys.stderr.flush()
            while not self._stop_event.wait(2.0):
                elapsed = time.perf_counter() - self._start_ts
                sys.stderr.write(f"[...] {self.label} ({elapsed:.1f}s)\n")
                sys.stderr.flush()
            return

        cycle = itertools.cycle(self.FRAMES)
        while not self._stop_event.is_set():
            frame = next(cycle)
            elapsed = time.perf_counter() - self._start_ts
            sys.stderr.write(f"\r[{frame}] {self.label} ({elapsed:.1f}s)   ")
            sys.stderr.flush()
            self._stop_event.wait(0.12)

    def update(self, new_label: str) -> None:
        self.label = new_label

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        elapsed = time.perf_counter() - self._start_ts
        status = "DONE" if exc_type is None else "FAILED"
        sys.stderr.write(f"\r[{status}] {self.label} ({elapsed:.2f}s)          \n")
        sys.stderr.flush()


class FallbackProgressBar:
    """Zero-dependency percentage progress bar fallback if `tqdm` is not installed."""

    def __init__(self, total: int, desc: str = "", unit: str = "step", disable: bool = False) -> None:
        self.total = max(1, int(total))
        self.n = 0
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.postfix = ""
        self._start = time.perf_counter()
        self._render()

    def set_description(self, desc: str) -> None:
        self.desc = desc
        self._render()

    def set_postfix_str(self, s: str) -> None:
        self.postfix = s
        self._render()

    def update(self, n: int = 1) -> None:
        self.n = min(self.total, self.n + n)
        self._render()

    def _render(self) -> None:
        if self.disable:
            return
        pct = (self.n / float(self.total)) * 100.0
        filled = int(24 * self.n // self.total)
        bar = "#" * filled + "-" * (24 - filled)
        elapsed = time.perf_counter() - self._start
        extra = f" | {self.postfix}" if self.postfix else ""
        sys.stderr.write(
            f"\r{self.desc}: {pct:5.1f}% |{bar}| {self.n}/{self.total} {self.unit} [{elapsed:.1f}s]{extra}   "
        )
        sys.stderr.flush()

    def close(self) -> None:
        if not self.disable:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def __enter__(self) -> "FallbackProgressBar":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def create_progress_bar(
    total: int,
    desc: str = "ArXiv-to-Code",
    unit: str = "step",
    disable: bool = False,
) -> Any:
    """Creates a live `tqdm` percentage progress bar (with automatic clean fallback)."""
    if _RealTqdm is not None:
        return _RealTqdm(
            total=max(1, int(total)),
            desc=desc,
            unit=unit,
            disable=disable,
            file=sys.stderr,
            ascii=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    return FallbackProgressBar(total=total, desc=desc, unit=unit, disable=disable)


@contextmanager
def spinner(label: str, enabled: bool = True) -> Iterator[ConsoleSpinner]:
    with ConsoleSpinner(label=label, enabled=enabled) as sp:
        yield sp
