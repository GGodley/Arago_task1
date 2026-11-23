"""Analyzer: figures and tables from sweep rows and single runs.

All figures are static PNGs for the report (matplotlib, Agg backend). Colors
follow one fixed assignment so a scheduler has the same color in every figure:

    fifo              blue
    greedy            orange
    memaware          aqua
    memaware-blocked  yellow (dashed)

Magnitudes (heatmaps) use a single-hue blue ramp. Runs that did not finish are
drawn in neutral gray with a hatch and a label, never as a colored value.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from .optimizer import CostModel, completed, group, pareto_front, recommend, scheduler_label  # noqa: E402

# Reference palette (light mode).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e0"
AXIS = "#c3c2b7"
NEUTRAL = "#d6d5ce"
SERIES = {
    "fifo": "#2a78d6",
    "greedy": "#eb6834",
    "memaware": "#1baf7a",
    "memaware-blocked": "#eda100",
    "memaware-snake": "#e87ba4",
    "greedy-blocked": "#6250d6",
}
LINESTYLE = {"memaware-blocked": (0, (5, 2)), "memaware-snake": (0, (1, 1.5)), "greedy-blocked": (0, (5, 2))}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SCHED_ORDER = ["fifo", "greedy", "memaware", "memaware-blocked", "memaware-snake", "greedy-blocked"]


def _style(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)


def _fig(w: float = 7, h: float = 4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _kcycles(c: float) -> str:
    if c >= 1e6:
        return f"{c / 1e6:.1f}M"
    if c >= 1e3:
        return f"{c / 1e3:.0f}k"
    return f"{c:.0f}"


# --------------------------------------------------------------------------
# Sweep figures
# --------------------------------------------------------------------------

def heatmap_units(rows: list[dict], workload: str, scheduler: str, mem: int, path: Path) -> Path | None:
    """Total cycles over Nm (x) x Nd (y) at one memory size."""
    sel = [r for r in rows if r["workload"] == workload and scheduler_label(r) == scheduler
           and r["local_mem_tiles"] == mem]
    if not sel:
        return None
    nds = sorted({r["n_dma"] for r in sel})
    nms = sorted({r["n_matrix"] for r in sel})
    grid = np.full((len(nds), len(nms)), np.nan)
    status = {}
    for r in sel:
        i, j = nds.index(r["n_dma"]), nms.index(r["n_matrix"])
        status[(i, j)] = r.get("status") or ("completed" if r["completed"] else "deadlock")
        if r["completed"]:
            grid[i, j] = r["total_cycles"]
    fig, ax = _fig(6.2, 4.6)
    ok = grid[~np.isnan(grid)]
    if ok.size:
        norm = LogNorm(vmin=ok.min(), vmax=max(ok.max(), ok.min() * 1.01))
        im = ax.imshow(grid, cmap=BLUES.reversed(), norm=norm, origin="lower", aspect="auto")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("total cycles (darker = faster)", color=TEXT_2, fontsize=9)
        cb.ax.tick_params(colors=TEXT_2, labelsize=8)
        cb.outline.set_visible(False)
    for (i, j), st in status.items():
        if st != "completed":
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=NEUTRAL, hatch="///",
                                       edgecolor=SURFACE, linewidth=0))
            ax.text(j, i, {"deadlock": "deadlock", "too_slow": ">20x LB", "infeasible": "n/a"}.get(st, st),
                    ha="center", va="center", fontsize=7, color=TEXT_2)
        else:
            v = grid[i, j]
            dark = ok.size and math.log(v) < (math.log(ok.min()) + math.log(ok.max())) / 2
            ax.text(j, i, _kcycles(v), ha="center", va="center", fontsize=8, color=SURFACE if dark else TEXT)
    ax.set_xticks(range(len(nms)), nms)
    ax.set_yticks(range(len(nds)), nds)
    _style(ax, f"{workload}, {scheduler}, memory {mem} tiles", "matrix units (Nm)", "DMA units (Nd)")
    ax.grid(False)
    return _save(fig, path)


def cycles_vs_memory(rows: list[dict], workload: str, nd: int, nm: int, path: Path,
                     schedulers: list[str] | None = None) -> Path | None:
    """Cycles against memory size, one line per scheduler, at fixed Nd and Nm."""
    sel = [r for r in rows if r["workload"] == workload and r["n_dma"] == nd and r["n_matrix"] == nm]
    if not sel:
        return None
    by = group(sel, "scheduler", "order")
    labels = [lab for lab in SCHED_ORDER if any(scheduler_label(v[0]) == lab for v in by.values())]
    if schedulers:
        labels = [lab for lab in labels if lab in schedulers]
    fig, ax = _fig()
    mems = sorted({r["local_mem_tiles"] for r in sel})
    top = max((r["total_cycles"] for r in completed(sel)), default=1)
    lb = min((r["lower_bound_ticks"] for r in sel if r.get("lower_bound_ticks")), default=None)
    ends: list[list] = []
    for lab in labels:
        pts = sorted((r for r in sel if scheduler_label(r) == lab), key=lambda r: r["local_mem_tiles"])
        xs = [r["local_mem_tiles"] for r in pts if r["completed"]]
        ys = [r["total_cycles"] for r in pts if r["completed"]]
        ax.plot(xs, ys, color=SERIES[lab], linewidth=2, linestyle=LINESTYLE.get(lab, "-"),
                marker="o", markersize=4, label=lab, zorder=3)
        fails = [r["local_mem_tiles"] for r in pts if not r["completed"]]
        if fails:
            ax.scatter(fails, [top * 1.6] * len(fails), marker="x", s=30, color=SERIES[lab], zorder=3)
        if xs:
            ends.append([lab, xs[-1], ys[-1]])
    if lb:
        ax.axhline(lb * 128, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=2)
        ax.annotate("lower bound", (mems[0], lb * 128), xytext=(2, 4), textcoords="offset points",
                    fontsize=8, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(mems, mems)
    ax.set_xlim(mems[0] / 1.3, mems[-1] * 1.3)
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    if any(not r["completed"] for r in sel if scheduler_label(r) in labels):
        ax.axhspan(top * 1.3, top * 2.0, color=GRID, zorder=1)
        ax.text(1.01, top * 1.6, "did not\nfinish (x)", transform=trans, fontsize=7, color=TEXT_2,
                va="center", ha="left")
    # End-of-line labels, spread apart in log space so they never collide.
    ends.sort(key=lambda e: e[2])
    ly = [math.log10(e[2]) for e in ends]
    lo, hi = ax.get_ylim()
    gap = 0.045 * (math.log10(hi) - math.log10(lo))
    for i in range(1, len(ly)):
        ly[i] = max(ly[i], ly[i - 1] + gap)
    for (lab, x, y), yy in zip(ends, ly):
        ax.annotate(lab, (x, y), xytext=(1.01, 10 ** yy), textcoords=trans, fontsize=8, color=TEXT_2,
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8, shrinkA=0, shrinkB=3))
    _style(ax, f"{workload}: cycles vs local memory (Nd={nd}, Nm={nm})", "local memory (tiles)", "total cycles")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=len(labels))
    return _save(fig, path)


def pareto_plot(rows: list[dict], workload: str, path: Path, model: CostModel = CostModel(),
                schedulers: tuple[str, ...] = ("fifo", "greedy", "memaware")) -> Path | None:
    """Cost vs cycles for every completed config; Pareto front per scheduler drawn as a step line."""
    sel = [r for r in rows if r["workload"] == workload and r["completed"]]
    if not sel:
        return None
    fig, ax = _fig()
    notes: list[str] = []
    for lab in schedulers:
        pts = [r for r in sel if scheduler_label(r) == lab]
        if not pts:
            continue
        ax.scatter([model.cost(r) for r in pts], [r["total_cycles"] for r in pts], s=10,
                   color=SERIES[lab], alpha=0.25, linewidths=0, zorder=2)
        front = pareto_front(pts, model)
        ax.plot([model.cost(r) for r in front], [r["total_cycles"] for r in front], color=SERIES[lab],
                linewidth=2, drawstyle="steps-post", marker="o", markersize=4, label=f"{lab} front", zorder=3)
        rec = recommend(pts, model)
        if rec:
            ax.scatter([model.cost(rec)], [rec["total_cycles"]], s=90, facecolor="none",
                       edgecolor=TEXT, linewidth=1.2, zorder=4)
            notes.append(f"{lab}: Nd={rec['n_dma']} Nm={rec['n_matrix']} mem={rec['local_mem_tiles']}"
                         f"  ({rec['total_cycles']:,} cyc, cost {model.cost(rec):.1f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(ax, f"{workload}: cost vs cycles (circled = recommended, within 10% of fastest)",
           f"relative cost = {model.matrix}·Nm + {model.dma}·Nd + {model.mem_tile}·mem", "total cycles")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="upper right")
    if notes:
        ax.text(0.02, 0.03, "recommended\n" + "\n".join(notes), transform=ax.transAxes, fontsize=7,
                color=TEXT_2, va="bottom", ha="left", family="monospace",
                bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.4"))
    return _save(fig, path)


def utilization_vs_units(rows: list[dict], workload: str, scheduler: str, mem: int, path: Path) -> Path | None:
    """Matrix utilization against Nm, one line per Nd."""
    sel = [r for r in rows if r["workload"] == workload and scheduler_label(r) == scheduler
           and r["local_mem_tiles"] == mem and r["completed"]]
    if not sel:
        return None
    fig, ax = _fig()
    nds = sorted({r["n_dma"] for r in sel})
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    for k, nd in enumerate(nds):
        pts = sorted((r for r in sel if r["n_dma"] == nd), key=lambda r: r["n_matrix"])
        color = ramp[min(k + (len(ramp) - len(nds)), len(ramp) - 1)]
        ax.plot([r["n_matrix"] for r in pts], [r["matrix_util"] * 100 for r in pts], color=color,
                linewidth=2, marker="o", markersize=4, label=f"Nd={nd}")
        ax.annotate(f"Nd={nd}", (pts[-1]["n_matrix"], pts[-1]["matrix_util"] * 100), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=TEXT_2, va="center")
    ax.set_xscale("log", base=2)
    nms = sorted({r["n_matrix"] for r in sel})
    ax.set_xticks(nms, nms)
    ax.set_ylim(0, 105)
    _style(ax, f"{workload}, {scheduler}, memory {mem}: matrix unit utilization", "matrix units (Nm)",
           "matrix utilization (%)")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, title="DMA units", title_fontsize=8)
    return _save(fig, path)


# --------------------------------------------------------------------------
# Single-run figures (need a SimResult recorded with record=True)
# --------------------------------------------------------------------------

def memory_timeline(results: list, path: Path, labels: list[str] | None = None) -> Path:
    """Local memory occupancy (% of capacity) over time, one step line per run."""
    fig, ax = _fig(7, 3.6)
    colors = list(SERIES.values())
    for k, r in enumerate(results):
        xs = [t * r.config.tick_cycles for t, _ in r.mem_series] + [r.total_cycles]
        ys = [u / r.config.local_mem_tiles * 100 for _, u in r.mem_series]
        ys = ys + [ys[-1]]
        lab = labels[k] if labels else r.scheduler
        ax.step(xs, ys, where="post", color=colors[k % len(colors)], linewidth=1.5, label=lab)
    ax.set_ylim(0, 105)
    r0 = results[0]
    _style(ax, f"{r0.workload}: local memory in use over time ({r0.config.label})", "cycle",
           "memory in use (% of capacity)")
    if len(results) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
    return _save(fig, path)


def gantt(result, path: Path, max_cycles: int | None = None) -> Path:
    """One row per unit; bars colored by command kind."""
    kind_color = {"LOAD": "#2a78d6", "COMPUTE": "#eb6834", "ACC": "#1baf7a", "SAVE": "#eda100"}
    cfg = result.config
    units = [("dma", k) for k in range(cfg.n_dma)] + [("matrix", k) for k in range(cfg.n_matrix)]
    row_of = {u: k for k, u in enumerate(units)}
    tc = result.config.tick_cycles
    fig, ax = _fig(8, 0.45 * len(units) + 1.2)
    for c in result.schedule:
        if max_cycles and c.start * tc > max_cycles:
            continue
        y = row_of[(c.unit, c.slot)]
        ax.barh(y, (c.end - c.start) * tc, left=c.start * tc, height=0.7, color=kind_color[c.kind],
                edgecolor=SURFACE, linewidth=1, hatch="///" if c.reload else None)
    ax.set_yticks(range(len(units)), [f"{u} {s}" for u, s in units])
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in kind_color.values()]
    labels = list(kind_color)
    if any(c.reload for c in result.schedule):
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=kind_color["LOAD"], hatch="///", edgecolor=SURFACE))
        labels.append("reload")
    ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=TEXT_2,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style(ax, f"{result.workload}: unit timeline ({result.config.label}, {result.scheduler})", "cycle")
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    return _save(fig, path)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def recommendations_table(rows: list[dict], model: CostModel = CostModel(), tolerance: float = 0.10) -> str:
    """Markdown: per workload and scheduler, the fastest config and the recommended one."""
    lines = ["| workload | scheduler | fastest (cycles, Nd/Nm/mem) | recommended (cycles, Nd/Nm/mem, cost) | finished |",
             "|---|---|---|---|---|"]
    by = group(rows, "workload")
    for wl in sorted(by, key=lambda w: (by[w][0]["M"] * by[w][0]["K"] * by[w][0]["N"], w)):
        sel = by[wl]
        name = wl[0] if isinstance(wl, tuple) else wl
        for lab in SCHED_ORDER:
            pts = [r for r in sel if scheduler_label(r) == lab]
            if not pts:
                continue
            ok = completed(pts)
            fin = f"{len(ok)}/{len(pts)}"
            if not ok:
                lines.append(f"| {name} | {lab} | none | none | {fin} |")
                continue
            fast = min(ok, key=lambda r: (r["total_cycles"], model.cost(r)))
            rec = recommend(pts, model, tolerance)
            lines.append(
                f"| {name} | {lab} | {fast['total_cycles']:,} ({fast['n_dma']}/{fast['n_matrix']}/{fast['local_mem_tiles']}) "
                f"| {rec['total_cycles']:,} ({rec['n_dma']}/{rec['n_matrix']}/{rec['local_mem_tiles']}, "
                f"{model.cost(rec):.2f}) | {fin} |")
    return "\n".join(lines)


def slices_plot(rows: list[dict], workload: str, center: tuple[int, int, int], path: Path) -> Path | None:
    """Three panels, one variable at a time around `center` (Nd, Nm, mem): cycles per scheduler,
    with the lower bound for each point as a dotted gray line."""
    sel = [r for r in rows if r["workload"] == workload]
    if not sel:
        return None
    d0, m0, t0 = center
    axes_def = [("n_dma", "DMA units (Nd)", lambda r: r["n_matrix"] == m0 and r["local_mem_tiles"] == t0),
                ("n_matrix", "matrix units (Nm)", lambda r: r["n_dma"] == d0 and r["local_mem_tiles"] == t0),
                ("local_mem_tiles", "local memory (tiles)", lambda r: r["n_dma"] == d0 and r["n_matrix"] == m0)]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    labels = [lab for lab in SCHED_ORDER if any(scheduler_label(r) == lab for r in sel)]
    for ax, (col, xlabel, keep) in zip(axs, axes_def):
        pts = [r for r in sel if keep(r)]
        xs_all = sorted({r[col] for r in pts})
        bound = {r[col]: r["lower_bound_ticks"] * 128 for r in pts if r.get("lower_bound_ticks")}
        if bound:
            bx = sorted(bound)
            ax.plot(bx, [bound[x] for x in bx], color=MUTED, linewidth=1, linestyle=(0, (2, 2)),
                    label="lower bound", zorder=2)
        for lab in labels:
            ok = sorted((r for r in pts if scheduler_label(r) == lab and r["completed"]), key=lambda r: r[col])
            if ok:
                ax.plot([r[col] for r in ok], [r["total_cycles"] for r in ok], color=SERIES[lab], linewidth=2,
                        linestyle=LINESTYLE.get(lab, "-"), marker="o", markersize=5, label=lab, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        if xs_all:
            ax.set_xticks(xs_all, xs_all)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        _style(ax, None, xlabel, "total cycles" if ax is axs[0] else None)
    fig.suptitle(f"{workload}: one variable at a time around Nd={d0}, Nm={m0}, mem={t0}",
                 x=0.01, ha="left", color=TEXT, fontsize=11)
    handles, labs = axs[2].get_legend_handles_labels()
    for ax in axs[:2]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labs:
                handles.append(h)
                labs.append(l)
    fig.legend(handles, labs, frameon=False, fontsize=8, labelcolor=TEXT_2, loc="lower center",
               ncol=len(labs), bbox_to_anchor=(0.5, 0.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
