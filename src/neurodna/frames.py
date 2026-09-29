"""Frame interpretation (static / ensemble / trajectory) and sampling summaries.

Times are in picoseconds (ps).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class FrameKind(Enum):
    """How the frames of a :class:`~neurodna.Complex` may be interpreted."""

    STATIC = "static"
    """A single structure. No dynamics can be inferred."""
    ENSEMBLE = "ensemble"
    """Several models with no time ordering (e.g. NMR models). No dynamics can be inferred."""
    TRAJECTORY = "trajectory"
    """Time-ordered frames from a simulation."""


class TimeSource(Enum):
    """Where trajectory frame times (ps) come from."""

    FILE = "file"
    """Times stored in the trajectory file (or in-memory reader), used as-is."""
    USER = "user"
    """``frame_index * timestep_ps`` from a user-supplied timestep."""
    UNAVAILABLE = "unavailable"
    """The format stores no time; time-dependent metrics are refused."""


@dataclass(frozen=True)
class FrameInfo:
    """Summary of the frames of a :class:`~neurodna.Complex`.

    Attributes:
        kind: Static structure, unordered ensemble, or time-ordered trajectory.
        n_frames: Number of frames (models).
        time_source: For trajectories, where frame times come from; ``None`` otherwise.
        timestep_ps: Nominal time between frames in ps, when known.
        has_box: Whether the first frame has periodic box dimensions (valid or not).
    """

    kind: FrameKind
    n_frames: int
    time_source: TimeSource | None
    timestep_ps: float | None
    has_box: bool

    @property
    def is_time_resolved(self) -> bool:
        return self.kind is FrameKind.TRAJECTORY

    @property
    def has_times(self) -> bool:
        """True for trajectories whose frame times are known."""
        return self.time_source in (TimeSource.FILE, TimeSource.USER)


def sampling_summary(frames: Sequence[int], times: Sequence[float]) -> dict[str, Any]:
    """Frame count, time span and sampling intervals (ps) of analysed frames.

    ``uniform_sampling`` is ``None`` when times are unknown or fewer than two
    frames were analysed.
    """
    t = np.asarray(times, dtype=float)
    out: dict[str, Any] = {
        "n_frames": len(frames),
        "first_frame": int(frames[0]) if len(frames) else None,
        "last_frame": int(frames[-1]) if len(frames) else None,
        "start_time_ps": float(t[0]) if t.size else math.nan,
        "end_time_ps": float(t[-1]) if t.size else math.nan,
        "sampling_interval_ps": math.nan,
        "min_sampling_interval_ps": math.nan,
        "max_sampling_interval_ps": math.nan,
        "uniform_sampling": None,
    }
    if t.size >= 2 and np.all(np.isfinite(t)):
        dt = np.diff(t)
        median = float(np.median(dt))
        out.update(
            sampling_interval_ps=median,
            min_sampling_interval_ps=float(dt.min()),
            max_sampling_interval_ps=float(dt.max()),
            uniform_sampling=bool(np.allclose(dt, median, rtol=1e-4, atol=1e-6)),
        )
    return out
