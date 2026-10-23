"""Assess saved simulation samples without importing Genesis.

Example:
  python four_view_genesis/assess_run.py four_view_genesis/outputs/trial_wrap

Only sampled, instantaneous geometry is assessed. An outcome requires two valid
sleeve crossings and at least four active seam pairs continuously in the SAVED
samples for one second after an observed bilateral grasp and release.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .validation import analyze_cloth
except ImportError:
    from validation import analyze_cloth


DEFAULT_GEOMETRY = {
    "arms": {
        "left": [[-.09, 0, 1.025], [-.15, -.008, .925]],
        "right": [[.09, 0, 1.025], [.15, -.008, .925]],
    },
    "torso_center": [0, 0, .91],
    "torso_radii": [.105, .083, .145],
    "provenance": "Assumed centerlines used by this reconstruction's simulate.py; not measured original physics.",
}


def evaluate_saved_samples(samples, minimum_active_seams=4, minimum_hold_s=1.0):
    """Conservative sampled hold logic, separated for analytic unit tests."""
    observed_bilateral_grasp = False
    released_previously = False
    release_time = None
    hold_start = None
    release_events = []
    numerical_failure = any(sample.get("numerical_state_plausible") is False for sample in samples)
    for sample in samples:
        grips = sample.get("grips_active")
        if grips is not None and len(grips) == 2 and all(grips):
            observed_bilateral_grasp = True
        released = bool(grips is not None and len(grips) == 2 and not any(grips))
        if observed_bilateral_grasp and released:
            if not released_previously:
                release_time = float(sample["time_s"])
                release_events.append(release_time)
            released_previously = True
        else:
            released_previously = False
            release_time = None
        seam_count = sample.get("active_seam_count")
        qualifying = bool(release_time is not None and sample["threading_passed"]
                          and sample.get("numerical_state_plausible", True)
                          and seam_count is not None and seam_count >= minimum_active_seams)
        if qualifying:
            if hold_start is None:
                hold_start = float(sample["time_s"])
        else:
            hold_start = None
    final = samples[-1] if samples else None
    duration = float(final["time_s"] - hold_start) if final is not None and hold_start is not None else 0.0
    success = bool(final is not None and release_time is not None and not numerical_failure
                   and duration + 1e-9 >= minimum_hold_s)
    reasons = []
    if numerical_failure:
        reasons.append("catastrophic_edge_stretch_seen_in_saved_trajectory")
    if not observed_bilateral_grasp:
        reasons.append("no_bilateral_grasp_observed")
    if release_time is None:
        reasons.append("no_final_release_episode_observed_after_bilateral_grasp")
    if final is None or not final["threading_passed"]:
        reasons.append("both_matching_sleeves_do_not_pass_at_final_sample")
    if final is None or final.get("active_seam_count") is None \
            or final["active_seam_count"] < minimum_active_seams:
        reasons.append("insufficient_active_seam_pairs_at_final_sample")
    if duration + 1e-9 < minimum_hold_s:
        reasons.append("insufficient_continuous_qualifying_saved_hold_after_release")
    return {
        "sampled_task_success": success,
        "failure_reasons": reasons,
        "observed_bilateral_grasp": observed_bilateral_grasp,
        "numerical_stretch_failure_observed": numerical_failure,
        "observed_release_times_s": release_events,
        "final_release_time_s": release_time,
        "final_qualifying_hold_start_s": hold_start,
        "final_qualifying_hold_duration_s": duration,
        "minimum_active_seam_pairs": int(minimum_active_seams),
        "minimum_after_release_hold_s": float(minimum_hold_s),
        "final_both_sleeves_passed": bool(final and final["threading_passed"]),
        "final_active_seam_count": final.get("active_seam_count") if final else None,
    }


def assess_run(output_directory, geometry=None, minimum_active_seams=4, minimum_hold_s=1.0):
    output_directory = Path(output_directory)
    if geometry is None:
        config_path=output_directory/'config.json'
        config=json.loads(config_path.read_text()) if config_path.exists() else {}
        geometry=config.get('validation_geometry',DEFAULT_GEOMETRY)
    with np.load(output_directory / "topology.npz", allow_pickle=False) as data:
        rest, faces = data["rest"], data["faces"]
    with np.load(output_directory / "trajectory.npz", allow_pickle=False) as data:
        positions, times = data["positions"], data["times"]
    if positions.ndim != 3 or len(positions) != len(times) or len(times) == 0 \
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("trajectory requires nonempty positions and strictly increasing finite times")
    metrics_path = output_directory / "metrics.json"
    history = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    history = sorted(history, key=lambda row: float(row["time"]))
    metric_times = np.array([float(row["time"]) for row in history])
    tolerance = max(1e-6, float(np.min(np.diff(times))) * .1) if len(times) > 1 else 1e-6
    samples = []
    for time, current in zip(times, positions):
        geometry_result = analyze_cloth(rest, faces, current, geometry["arms"],
                                        geometry["torso_center"], geometry["torso_radii"])
        metric = None
        if len(metric_times):
            index = int(np.argmin(np.abs(metric_times - time)))
            if abs(float(metric_times[index] - time)) <= tolerance:
                metric = history[index]
        seam = metric.get("velcro", {}) if metric else {}
        sleeves = {}
        for side, sleeve in geometry_result["sleeves"].items():
            sleeves[side] = {
                key: sleeve[key] for key in (
                    "passed", "reliable", "unreliable_reasons", "center", "normal",
                    "planarity_rms_m", "planarity_max_m", "planarity_rms_over_diameter",
                    "projected_area_m2", "projected_diameter_m", "arm_crossings")}
        ratios = geometry_result["edge_stretch"]["length_ratio"]
        samples.append({
            "time_s": float(time),
            "stage": metric.get("stage") if metric else None,
            "metrics_matched": metric is not None,
            "grips_active": [bool(value) for value in metric["grips_active"]]
            if metric and "grips_active" in metric else None,
            "threading_passed": geometry_result["threading_passed"],
            "geometry_reliable": geometry_result["reliable"],
            "topology_passed": geometry_result["topology"]["passed"],
            "topology_error": geometry_result["topology"]["error"],
            "sleeves": sleeves,
            "max_torso_radial_penetration_m": geometry_result["torso_penetration"]["max_radial_depth_m"],
            "inside_torso_vertex_count": geometry_result["torso_penetration"]["inside_vertex_count"],
            "p99_edge_length_ratio": ratios["p99"] if ratios else None,
            "max_edge_length_ratio": ratios["max"] if ratios else None,
            "numerical_state_plausible": bool(ratios and ratios["p99"] <= 3.0 and ratios["max"] <= 10.0),
            "active_seam_count": int(seam["active_count"]) if "active_count" in seam else None,
            "mean_active_seam_gap_m": seam.get("mean_active_gap_m"),
            "max_active_seam_gap_m": seam.get("max_active_gap_m"),
        })
    passing = [sample["time_s"] for sample in samples if sample["threading_passed"]]
    stretch = [sample["p99_edge_length_ratio"] for sample in samples if sample["p99_edge_length_ratio"] is not None]
    outcome = evaluate_saved_samples(samples, minimum_active_seams, minimum_hold_s)
    return {
        "scope": "Instantaneous checks at saved samples only. Inter-sample collisions, triangle-interior penetration, and full arm volume clearance are not certified.",
        "success_scope": "sampled_task_success means matching sleeve crossings and active seam count persist in saved samples for the required hold after an observed grasp/release; it is not recovery of the unknown original simulation.",
        "input_directory": str(output_directory.resolve()),
        "validation_geometry": geometry,
        "sample_count": len(samples),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "max_saved_sample_interval_s": float(np.max(np.diff(times))) if len(times) > 1 else None,
        "missing_metric_sample_count": sum(not sample["metrics_matched"] for sample in samples),
        "first_both_sleeves_passed_time_s": passing[0] if passing else None,
        "both_sleeves_passed_sample_count": len(passing),
        "max_torso_radial_penetration_m": max(sample["max_torso_radial_penetration_m"] for sample in samples),
        "max_p99_edge_length_ratio": max(stretch) if stretch else None,
        "catastrophic_stretch_sanity_limits": {
            "p99_edge_length_ratio": 3.0, "max_edge_length_ratio": 10.0,
            "meaning": "Loose numerical explosion guard only, not a calibrated material acceptance criterion"},
        "outcome": outcome,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--geometry", type=Path, help="JSON containing arms, torso_center, torso_radii")
    parser.add_argument("--minimum-active-seams", type=int, default=4)
    parser.add_argument("--minimum-hold", type=float, default=1.0)
    args = parser.parse_args()
    if args.minimum_active_seams < 4 or args.minimum_hold < 1.0:
        parser.error("success thresholds may be tightened but not lowered below 4 seam pairs / 1 second")
    geometry = json.loads(args.geometry.read_text()) if args.geometry else None
    result = assess_run(args.output_directory, geometry, args.minimum_active_seams, args.minimum_hold)
    target = args.output_directory / "assessment.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"assessment": str(target.resolve()), "sample_count": result["sample_count"],
                      "first_both_sleeves_passed_time_s": result["first_both_sleeves_passed_time_s"],
                      "outcome": result["outcome"]}, indent=2))


if __name__ == "__main__":
    main()
