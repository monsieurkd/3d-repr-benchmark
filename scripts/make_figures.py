"""Generate the 4 blog figures for the 3D-representation post.

Output PNGs (transparent background, dual light/dark theme safe):
  personal-website/public/blog/3d-rep/fig-reps.png
  personal-website/public/blog/3d-rep/fig-training-time.png
  personal-website/public/blog/3d-rep/fig-throttle.png
  personal-website/public/blog/3d-rep/fig-flops-vs-wallclock.png

Palette (validated, see dataviz skill; passes CVD + normal-vision floors on both
#ffffff and #111114): one fixed hue per representation, used identically in every
figure so colour == identity. Ink is neutral mid-gray so it reads on light and
dark transparent backgrounds alike.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import convert as cv  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
OUT = "/Users/kieuminhduc/dev/personal-website/public/blog/3d-rep"

# --- palette: fixed hue per representation, identical across every figure ---------
COL = {
    "point":  "#3987e5",   # blue   (slot 1)
    "voxel":  "#d95926",   # orange (slot 2)
    "sparse": "#199e70",   # aqua   (slot 3)
    "mview":  "#c98500",   # yellow (slot 4)
}
LABEL = {
    "point":  "Point cloud",
    "voxel":  "Dense voxel",
    "sparse": "Sparse voxel",
    "mview":  "Multi-view 2D",
}
ORDER = ["mview", "sparse", "point", "voxel"]   # consistent narrative order

# Neutral ink that reads on a transparent PNG in both light and dark themes.
INK = "#525252"
INK_LIGHT = "#8a8a8a"
GRID = (0.55, 0.55, 0.55, 0.18)

plt.rcParams.update({
    "font.family": ["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
    "font.size": 12,
    "text.color": INK,
    "axes.edgecolor": INK_LIGHT,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})


def load(rep, suffix=""):
    with open(os.path.join(RESULTS, f"{rep}_all_c40_e15{suffix}.json")) as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", path)


# -------------------------------------------------------------------- fig: reps
def fig_reps():
    """4-panel: one real ModelNet mesh shown as point cloud / voxel /
    multi-view silhouettes (and a sparse-dot panel). Same object, four ways."""
    mesh = cv.normalize_mesh(cv.load_mesh(
        os.path.join(REPO, "data/ModelNet40/airplane/train/airplane_0001.off")))
    pc = cv.sample_points(mesh, 1024)
    vx = cv.voxelize_dense(mesh, 32)
    sp = cv.to_sparse_voxel_points(mesh, 32, 1024)
    mv = cv.to_multiview(mesh, 8, 64)

    fig = plt.figure(figsize=(12, 3.3))
    # a) point cloud (3D scatter)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=4, c=COL["point"], alpha=0.7, edgecolors="none")
    ax1.set_title("Point cloud\n1,024 surface dots", fontsize=11, color=INK, pad=8)
    _clean3d(ax1)

    # b) dense voxel: occupied cells as 3D scatter, cube outline
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    occ = np.argwhere(vx)
    ax2.scatter(occ[:, 2], occ[:, 1], occ[:, 0], s=2.5, c=COL["voxel"], alpha=0.55, edgecolors="none")
    ax2.set_title(f"Dense voxel\n32x32x32 cube, {len(occ)} filled", fontsize=11, color=INK, pad=8)
    _clean3d(ax2)

    # c) sparse voxel: occupied-cell centres as points
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax3.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=4, c=COL["sparse"], alpha=0.7, edgecolors="none")
    ax3.set_title("Sparse voxel\nonly filled cells", fontsize=11, color=INK, pad=8)
    _clean3d(ax3)

    # d) multi-view: 8 silhouettes in a 2x4 contact sheet
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_axis_off()
    grid = np.zeros((64 * 2 + 6, 64 * 4 + 18))
    for i in range(8):
        r, c = divmod(i, 4)
        grid[r * 64 + r * 6: r * 64 + r * 6 + 64,
             c * 64 + c * 6: c * 64 + c * 6 + 64] = mv[i]
    ax4.imshow(grid, cmap="gray_r", aspect="equal")
    ax4.set_title("Multi-view 2D\n8 silhouettes @ 64x64", fontsize=11, color=INK, pad=8)

    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig-reps.png")


def _clean3d(ax):
    """Minimal 3D axes: no panes, no ticks, thin neutral frame."""
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_visible(False)
        a.line.set_color(INK_LIGHT)
        a._axinfo["grid"]["color"] = (0.6, 0.6, 0.6, 0.25)
        a._axinfo["grid"]["linewidth"] = 0.5
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.tick_params(colors=INK_LIGHT)


# ------------------------------------------------------------ fig: training time
def fig_training_time():
    """Horizontal bars: total training time (min) for the 4 reps, fastest on top."""
    reps = ORDER[::-1]  # so fastest (mview) sits at top after barh
    mins = [load(r)["total_train_sec"] / 60 for r in reps]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(reps))
    bars = ax.barh(y, mins, height=0.55, color=[COL[r] for r in reps],
                   edgecolor="none", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[r] for r in reps], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Total training time for 15 epochs (minutes)", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(0, max(mins) * 1.18)
    for b, m in zip(bars, mins):
        ax.text(b.get_width() + max(mins) * 0.015, b.get_y() + b.get_height() / 2,
                f"{m:.1f} min", va="center", ha="left", color=INK, fontsize=11)
    save(fig, "fig-training-time.png")


# ----------------------------------------------------------------- fig: throttle
def fig_throttle():
    """Per-epoch time: dense voxel climbs within a run, multi-view stays flat.
    Uses the two voxel runs to also show run-to-run spread."""
    mview = load("mview")["per_epoch_sec"]
    voxel = load("voxel")["per_epoch_sec"]
    voxel2 = load("voxel", ".throttled")["per_epoch_sec"]
    epochs = np.arange(1, len(mview) + 1)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(epochs, voxel, color=COL["voxel"], lw=2, marker="o", ms=4,
            markerfacecolor=COL["voxel"], markeredgecolor="none", label="Dense voxel (run 1)")
    ax.plot(epochs, voxel2, color=COL["voxel"], lw=1.4, ls=":",
            marker="o", ms=3.5, markerfacecolor=COL["voxel"], markeredgecolor="none",
            alpha=0.7, label="Dense voxel (run 2)")
    ax.plot(epochs, mview, color=COL["mview"], lw=2, marker="o", ms=4,
            markerfacecolor=COL["mview"], markeredgecolor="none", label="Multi-view 2D")

    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("Seconds per epoch", color=INK)
    ax.set_xticks(epochs[::2])
    ax.grid(color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    leg = ax.legend(frameon=False, fontsize=10, loc="upper left",
                    labelcolor=INK)
    for t in leg.get_texts():
        t.set_color(INK)
    # annotate the climb on voxel run 1
    ax.annotate("dense voxel climbs\nwithin a single run",
                xy=(15, voxel[-1]), xytext=(9.5, voxel[-1] + 8),
                color=INK_LIGHT, fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK_LIGHT, lw=0.7))
    save(fig, "fig-throttle.png")


# ----------------------------------------------------- fig: flops vs wall-clock
def fig_flops_vs_wallclock():
    """The proof figure: FLOPs ordering and wall-clock ordering disagree.
    Two side-by-side horizontal bar charts, same reps, same colours."""
    with open(os.path.join(RESULTS, "flops.json")) as f:
        fl = {r["rep"]: r for r in json.load(f)["runs"]}

    # wall-clock = total training minutes (main run)
    reps = ORDER[::-1]
    macs = [fl[r]["macs_per_sample"] / 1e6 for r in reps]      # millions of MACs
    mins = [load(r)["total_train_sec"] / 60 for r in reps]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    y = np.arange(len(reps))
    colors = [COL[r] for r in reps]

    axL.barh(y, macs, height=0.55, color=colors, edgecolor="none", zorder=3)
    axL.set_yticks(y)
    axL.set_yticklabels([LABEL[r] for r in reps], fontsize=11)
    axL.invert_yaxis()
    axL.set_title("FLOPs per sample\n(the theory)", color=INK, fontsize=12, pad=10)
    axL.set_xlabel("Multiply-accumulates (millions)", color=INK)
    for i, v in enumerate(macs):
        axL.text(v + max(macs) * 0.02, i, f"{v:.0f}M", va="center", color=INK, fontsize=10)

    axR.barh(y, mins, height=0.55, color=colors, edgecolor="none", zorder=3)
    axR.set_title("Wall-clock to train\n(the reality)", color=INK, fontsize=12, pad=10)
    axR.set_xlabel("Minutes for 15 epochs", color=INK)
    for i, v in enumerate(mins):
        axR.text(v + max(mins) * 0.02, i, f"{v:.1f} min", va="center", color=INK, fontsize=10)

    for ax in (axL, axR):
        ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.22)

    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig-flops-vs-wallclock.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_reps()
    fig_training_time()
    fig_throttle()
    fig_flops_vs_wallclock()
    print("\nAll 4 figures generated.")
