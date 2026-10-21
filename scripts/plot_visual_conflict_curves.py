"""Render the frozen Development perturbation sweep without fitting or smoothing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence/active_development_contract/stress_curves/visual_conflict_sweep.csv"
MANIFEST = DATA.with_name("visual_conflict_sweep_manifest.json")
OUTPUT = ROOT / "figures/quantitative/fig06_visual_conflict_curves"
QA = ROOT / "review/stress_curves"
PANELS = (
    ("bearing_bias", "Bearing bias", "Bias (deg)", 1.0, [0, 1, 2, 3, 4, 5, 6]),
    ("temporal_lag", "Temporal lag", "Lag (frames)", 1.0, [0, 2, 4, 8]),
    ("visual_dropout", "Frame dropout", "Missing evidence (%)", 100.0, [0, 25, 50, 75]),
)
COLORS = {"guard_off": "#697C8B", "protected": "#CA7D22", "radar": "#444444"}


def load_data() -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert digest == manifest["output_sha256"], "Frozen source checksum mismatch"
    frame = pd.read_csv(DATA)
    assert len(frame) == manifest["rows"] == 30
    assert not frame.isna().any().any()
    assert set(frame["frames"]) == {9969}
    assert set(frame["method"]) == {"guard_off", "protected"}
    assert set(frame["perturbation"]) == {row[0] for row in PANELS}
    assert not frame.duplicated(["perturbation", "severity", "method"]).any()
    assert frame["radar_ap_0p30m"].nunique() == 1
    for key, _, _, scale, ticks in PANELS:
        for method in ("guard_off", "protected"):
            rows = frame[(frame.perturbation == key) & (frame.method == method)]
            np.testing.assert_allclose(sorted(rows.severity * scale), ticks)
    for method in ("guard_off", "protected"):
        clean = frame[(frame.severity == 0) & (frame.method == method)]
        for field in ("ap_0p30m", "harmed_frame_rate", "visual_admission_rate"):
            assert clean[field].nunique() == 1, "Different clean references in sweep"
    for field in ("ap_0p30m", "harmed_frame_rate", "visual_admission_rate"):
        assert frame[field].between(0, 1).all()
    return frame, {"source_sha256": digest, "manifest_guard": manifest["guard"]}


def plot() -> None:
    frame, audit = load_data()
    QA.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.25), sharex="col", sharey="row")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.15, top=0.84,
                        wspace=0.20, hspace=0.30)
    plotted = []
    for col, (key, title, xlabel, scale, ticks) in enumerate(PANELS):
        for row, field in enumerate(("ap_0p30m", "harmed_frame_rate")):
            ax = axes[row, col]
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E6E8EB", linewidth=0.45)
            for method in ("guard_off", "protected"):
                subset = frame[(frame.perturbation == key) & (frame.method == method)]
                subset = subset.sort_values("severity")
                x = subset.severity.to_numpy() * scale
                y = subset[field].to_numpy() * (100 if row else 1)
                ax.plot(x, y, color=COLORS[method], linewidth=1.45,
                        linestyle="--" if method == "guard_off" else "-",
                        marker="s" if method == "guard_off" else "o",
                        markersize=3.4, markeredgewidth=0.8,
                        markerfacecolor="white" if method == "guard_off" else COLORS[method],
                        zorder=3 if method == "protected" else 2)
                plotted.extend({"panel": chr(97 + row * 3 + col), "method": method,
                                "perturbation": key, "x": float(xi), "metric": field,
                                "y": float(yi)} for xi, yi in zip(x, y))
            ax.set_xticks(ticks)
            span = ticks[-1] - ticks[0]
            ax.set_xlim(ticks[0] - 0.04 * span, ticks[-1] + 0.04 * span)
            ax.annotate(chr(97 + row * 3 + col), xy=(0, 1), xycoords="axes fraction",
                        xytext=(0, 7), textcoords="offset points",
                        ha="left", va="bottom", weight="bold", fontsize=8)
            if row == 0:
                ax.axhline(frame.radar_ap_0p30m.iloc[0], color=COLORS["radar"],
                           linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
                ax.set_ylim(0.58, 0.81)
                ax.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80])
                ax.set_title(title, pad=9, weight="normal")
            else:
                ax.set_ylim(-1, 50)
                ax.set_yticks([0, 10, 20, 30, 40, 50])
                ax.set_xlabel(xlabel, labelpad=4)
    axes[0, 0].set_ylabel("AP@0.30", labelpad=6)
    axes[1, 0].set_ylabel("Harmed frames (%)", labelpad=6)
    handles = [
        Line2D([], [], color=COLORS["guard_off"], linewidth=1.45, linestyle="--",
               marker="s", markersize=3.4, markerfacecolor="white", label="Guard off"),
        Line2D([], [], color=COLORS["protected"], linewidth=1.45, marker="o",
               markersize=3.4, label="Protected"),
        Line2D([], [], color=COLORS["radar"], linewidth=0.9, linestyle=(0, (2, 2)),
               label="Expert-radar (AP)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.995),
               ncol=3, frameon=False, handlelength=2.4, columnspacing=2.4,
               handletextpad=0.65, borderaxespad=0)
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig, json_out=QA / "alignment.json", overlay_svg=QA / "alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True,
    )
    # Fixed canvas preserves the audited physical panel geometry across exports.
    fig.savefig(OUTPUT.with_suffix(".pdf"), format="pdf")
    fig.savefig(OUTPUT.with_suffix(".svg"), format="svg")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    pd.DataFrame(plotted).to_csv(QA / "plotted_points.csv", index=False)
    audit.update({
        "input_rows": len(frame), "excluded_rows": 0, "drawn_points": len(plotted),
        "scored_frames_per_point": 9969, "split": "Development",
        "figure_size_mm": [177.8, 82.55], "interpolation": "straight segments only",
        "harm_percent_transform": "100 * harmed_frame_rate; all scored frames",
        "uncertainty": "Single frozen sweep; no replicate intervals are available or drawn",
        "alignment_verdict": alignment["verdict"],
    })
    (QA / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    plot()
