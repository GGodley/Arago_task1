"""Named workload presets used in the experiments and the report.

Classes are defined against the hardware (see README, "Workload classes"):

    small   fits in local memory, few output tiles: latency-bound
    medium  around the local memory sizes we sweep: the regime flips
    large   far beyond local memory: bandwidth- and reuse-bound
    skewed  shapes that stress one dimension
"""

from __future__ import annotations

from .translator import MatmulSpec

PRESETS: dict[str, list[MatmulSpec]] = {
    "small": [
        MatmulSpec(64, 64, 64, "small_64"),
        MatmulSpec(128, 128, 128, "small_128"),
        MatmulSpec(256, 256, 256, "small_256"),
    ],
    "medium": [
        MatmulSpec(512, 512, 512, "medium_512"),
        MatmulSpec(1024, 1024, 1024, "medium_1024"),
    ],
    "large": [
        MatmulSpec(2048, 2048, 2048, "large_2048"),
        MatmulSpec(4096, 4096, 4096, "large_4096"),
    ],
    "skewed": [
        MatmulSpec(64, 4096, 4096, "skew_decode_64x4096x4096"),   # one row of output tiles
        MatmulSpec(4096, 64, 4096, "skew_k1_4096x64x4096"),      # Kt = 1: no ACC
        MatmulSpec(64, 8192, 64, "skew_longk_64x8192x64"),       # long ACC chain
        MatmulSpec(100, 200, 300, "skew_padded_100x200x300"),    # not multiples of 64
    ],
}


def all_presets() -> dict[str, MatmulSpec]:
    """Flat name -> spec map over every class."""
    return {spec.label: spec for specs in PRESETS.values() for spec in specs}


def get_preset(name: str) -> list[MatmulSpec]:
    """A class name ('small', ...) or a single preset name ('medium_512')."""
    if name in PRESETS:
        return PRESETS[name]
    flat = all_presets()
    if name in flat:
        return [flat[name]]
    raise KeyError(f"unknown preset {name!r}; classes: {sorted(PRESETS)}, presets: {sorted(flat)}")
