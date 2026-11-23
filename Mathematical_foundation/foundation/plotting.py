"""Shared figure style so every experiment's graphs look the same."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, fixed order; greys for context and text.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREYS = ("#8a8985", "#b5b4ae", "#6f6e6a")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def new_figure():
    return plt.subplots(figsize=(9, 5.5), dpi=150)


def frame(ax, title: str, subtitle: str, xlabel: str, ylabel: str) -> None:
    """Recessive axes and grid, finding-first title, subtitle with units."""
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.minorticks_off()
    ax.set_xlabel(xlabel, color=INK2, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2, fontsize=11)
    ax.set_title(title, loc="left", color=INK, fontsize=14, pad=26)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=10)


def log2_size_axis(ax, sizes: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 2.6)  # room for end labels


def label_end(ax, x, y, text: str, color: str, dy: float = 0) -> None:
    """Direct label at the end of a series."""
    ax.plot([x], [y], "o", color=color, markersize=6)
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", color=INK, fontsize=10, va="center")


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
