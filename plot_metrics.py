"""Plot saved-sample validation for a completed reconstruction run.

Usage: python four_view_genesis/plot_metrics.py four_view_genesis/outputs/wrap_v6
Requires assessment.json from assess_run.py and the simulation's metrics.json.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

_CACHE = Path(__file__).resolve().parent / ".cache" / "matplotlib"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def plot_metrics(output_directory, destination=None):
    """Return the path to a three-panel PNG; do not alter assessment or inputs."""
    directory = Path(output_directory)
    assessment = json.loads((directory / "assessment.json").read_text())
    history = json.loads((directory / "metrics.json").read_text())
    samples = assessment["samples"]
    if not samples:
        raise ValueError("assessment has no saved samples")
    times = np.asarray([sample["time_s"] for sample in samples], dtype=float)
    outcome = assessment["outcome"]
    success = bool(outcome["sampled_task_success"])
    release_times = [float(time) for time in outcome.get("observed_release_times_s", [])]
    history = sorted(history, key=lambda row: float(row["time"]))
    seam_times = np.asarray([row["time"] for row in history], dtype=float)
    seam_counts = np.asarray([row.get("velcro", {}).get("active_count", np.nan)
                              for row in history], dtype=float)
    edge_ratios = np.asarray([sample["p99_edge_length_ratio"]
                              if sample["p99_edge_length_ratio"] is not None else np.nan
                              for sample in samples], dtype=float)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#B8C1CC", "xtick.color": "#475569",
                         "ytick.color": "#475569", "text.color": "#172033",
                         "axes.labelcolor": "#172033"})
    figure, axes = plt.subplots(3, 1, figsize=(12, 7.7), sharex=True,
                                gridspec_kw={"height_ratios": [1.05, 1, 1]})
    figure.subplots_adjust(left=.13, right=.97, top=.815, bottom=.135, hspace=.40)
    figure.suptitle(f"{directory.name}  |  Task success: {str(success).upper()}",
                    x=.13, y=.965, ha="left", fontsize=18, fontweight="bold",
                    color="#16784D" if success else "#A83932")
    requirement = (f"Required: both sleeves pass, at least {outcome['minimum_active_seam_pairs']} active seam pairs, "
                   f"and {outcome['minimum_after_release_hold_s']:g} s of hold after release.")
    figure.text(.13, .915, requirement, fontsize=10.5, color="#475569")
    figure.text(.13, .875,
                f"{len(samples)} saved samples  |  "
                f"Final seam pairs: {outcome.get('final_active_seam_count', 'unknown')}  |  "
                f"Qualifying hold after release: {outcome['final_qualifying_hold_duration_s']:.1f} s",
                fontsize=10, color="#475569")

    top = axes[0]
    lane_labels = ("Right sleeve", "Left sleeve", "Both sleeves")
    statuses = (
        [sample.get("sleeves", {}).get("right", {}).get("passed", False) for sample in samples],
        [sample.get("sleeves", {}).get("left", {}).get("passed", False) for sample in samples],
        [sample["threading_passed"] for sample in samples],
    )
    for lane, status in enumerate(statuses):
        passed = np.asarray(status, dtype=bool)
        y = lane + .5
        top.axhspan(lane + .16, lane + .84, color="#F2F5F8", zorder=0)
        top.scatter(times[~passed], np.full(np.count_nonzero(~passed), y), marker="s",
                    s=13, color="#C7CED7", linewidths=0, zorder=2)
        top.scatter(times[passed], np.full(np.count_nonzero(passed), y), marker="s",
                    s=22 if lane == 2 else 17, color="#16865B", linewidths=0, zorder=3)
    top.set_yticks([.5, 1.5, 2.5], lane_labels)
    top.set_ylim(0, 3.25)
    top.set_title("Sleeve passage at saved samples", loc="left", pad=9)
    top.spines["left"].set_visible(False)
    top.tick_params(axis="y", length=0)
    top.legend(handles=[
        Line2D([], [], color="#16865B", marker="s", linestyle="None", markersize=6, label="Passed"),
        Line2D([], [], color="#C7CED7", marker="s", linestyle="None", markersize=6,
               label="Failed or unreliable")],
        loc="upper right", bbox_to_anchor=(1, 1.28), ncol=2, frameon=False, fontsize=9)

    middle = axes[1]
    middle.step(seam_times, seam_counts, where="post", color="#176DA5", linewidth=1.8)
    middle.scatter(seam_times, seam_counts, color="#176DA5", s=5, zorder=3)
    middle.axhline(outcome["minimum_active_seam_pairs"], color="#A86617", linestyle=":", linewidth=1.4)
    middle.text(times[0] + (times[-1] - times[0]) * .015,
                outcome["minimum_active_seam_pairs"] + .25,
                f"Required minimum: {outcome['minimum_active_seam_pairs']}",
                color="#8B571A", fontsize=9)
    middle.set_title("Active hook-and-loop seam pairs", loc="left", pad=8)
    middle.set_ylabel("Pair count")
    finite_counts = seam_counts[np.isfinite(seam_counts)]
    middle.set_ylim(-.5, max(float(np.max(finite_counts)) if len(finite_counts) else 0,
                            float(outcome["minimum_active_seam_pairs"])) + 2)
    middle.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    middle.grid(axis="y", color="#E6EBF1", linewidth=.7)

    bottom = axes[2]
    bottom.plot(times, edge_ratios, color="#9A5D16", linewidth=1.8)
    bottom.scatter(times, edge_ratios, color="#9A5D16", s=5)
    bottom.axhline(1.0, color="#8A96A6", linestyle=":", linewidth=1.2)
    bottom.set_title("Cloth edge stretch: 99th percentile", loc="left", pad=8)
    bottom.set_ylabel("Length / initial length")
    bottom.set_xlabel("Simulation time (s)")
    finite_ratios = edge_ratios[np.isfinite(edge_ratios)]
    if len(finite_ratios):
        high = max(1.10, float(np.max(finite_ratios)) * 1.035)
        if high > 4:
            bottom.set_yscale("log")
            bottom.set_ylabel("Length / initial length (log)")
            bottom.set_ylim(max(.01, min(.95, float(np.min(finite_ratios)) * .95)), high)
        else:
            bottom.set_ylim(min(.99, float(np.min(finite_ratios)) - .01), high)
    bottom.grid(axis="y", color="#E6EBF1", linewidth=.7)
    span = max(float(times[-1] - times[0]), 1.0)
    bottom.set_xlim(float(times[0]) - span * .01, float(times[-1]) + span * .01)
    for release in release_times:
        for axis in axes:
            axis.axvline(release, color="#B74B44", linestyle="--", linewidth=1.15, zorder=4)
        top.annotate(f"Grippers released\n{release:g} s", (release, 2.92),
                     xytext=(-6, 2), textcoords="offset points", ha="right", va="bottom",
                     color="#A83932", fontsize=8.5)
    if not release_times:
        top.text(.99, .93, "No release observed", transform=top.transAxes, ha="right",
                 color="#A83932", fontsize=8.5)
    interval = assessment.get("max_saved_sample_interval_s")
    interval_text = f" Maximum saved interval: {interval:g} s." if interval is not None else ""
    figure.text(.13, .066,
                "Saved-sample validation only; this does not establish continuous passage or collision-free motion."
                + interval_text, fontsize=9, color="#475569")
    figure.text(.13, .038,
                "Pass criteria are unchanged. A closed seam alone does not establish successful dressing.",
                fontsize=9, color="#475569")
    destination = Path(destination) if destination else directory / "validation_timeline.png"
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--output", type=Path, help="PNG destination; defaults to validation_timeline.png in the run")
    args = parser.parse_args()
    destination = plot_metrics(args.output_directory, args.output)
    print(destination.resolve())


if __name__ == "__main__":
    main()
