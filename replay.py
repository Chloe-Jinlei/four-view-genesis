"""Render previously saved Genesis states without advancing the simulation.

This is recorded-state playback of an approximate reconstruction. Cloth particle
positions and robot joint coordinates are loaded from trajectory.npz; there are
no dynamics steps, invented intermediate states, or new physical predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import runtime
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL = "Saved Genesis states / approximate reconstruction"


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def resolve_run(value):
    supplied = Path(value).expanduser()
    if supplied.exists():
        return supplied.resolve()
    fallback = runtime.ROOT / "outputs" / supplied
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Run directory does not exist: {supplied}")


def load_recording(run_dir):
    config = json.loads((run_dir / "config.json").read_text())
    with np.load(run_dir / "trajectory.npz", allow_pickle=False) as data:
        positions = np.asarray(data["positions"])
        joints = np.asarray(data["joints"])
        times = np.asarray(data["times"], dtype=float)
    with np.load(run_dir / "topology.npz", allow_pickle=False) as data:
        rest = np.asarray(data["rest"])
        faces = np.asarray(data["faces"])
    if times.ndim != 1 or len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Recording requires at least two strictly increasing timestamps")
    if positions.shape != (len(times), len(rest), 3) or rest.shape != (len(rest), 3):
        raise ValueError("Cloth recording dimensions do not match saved rest topology")
    if joints.ndim != 3 or joints.shape[:2] != (len(times), 2):
        raise ValueError("Recording must contain both robots at every timestamp")
    if not all(np.isfinite(x).all() for x in (positions, joints, times, rest)):
        raise ValueError("Recording contains non-finite values")
    intervals = np.diff(times)
    frame_dt = float(np.median(intervals))
    if not np.allclose(intervals, frame_dt, atol=1e-7, rtol=1e-5):
        raise ValueError("Nonuniform timestamps require a variable frame-rate export; refusing to alter timing")
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        metric_times = np.array([record["time"] for record in metrics], dtype=float)
        stages = []
        for t in times:
            idx = int(np.argmin(abs(metric_times-t)))
            if abs(metric_times[idx]-t) > 1e-6:
                raise ValueError(f"Missing saved stage label at t={t}")
            stages.append(str(metrics[idx]["stage"]))
        stage_source = "metrics.json"
    else:
        # Match the stage annotation convention used when the states were saved.
        keys = config["trajectory"]
        key_times = [item["time"] for item in keys]
        stages = []
        for t in times:
            index = min(max(int(np.searchsorted(key_times, t, side="right")-1), 0), len(keys)-2)
            stages.append(str(keys[index+1]["stage"]))
        stage_source = "config.json trajectory (metrics.json absent)"
    return config, positions, joints, times, rest, faces, stages, frame_dt, stage_source


def contact_indices(stages, count=8):
    # Preserve initial and final states and distribute the other panels over the
    # actual stage runs. No screenshot represents an interpolated cloth state.
    boundaries = [0]
    boundaries += [i for i in range(1, len(stages)) if stages[i] != stages[i-1]]
    boundaries.append(len(stages))
    candidates = [0]
    candidates += [(start+stop-1)//2 for start, stop in zip(boundaries[:-1], boundaries[1:])]
    candidates.append(len(stages)-1)
    candidates = sorted(set(candidates))
    if len(candidates) > count:
        keep = np.rint(np.linspace(0, len(candidates)-1, count)).astype(int)
        candidates = [candidates[i] for i in keep]
    return candidates


def compose_frame(cameras, first_view, second_view, timestamp, stage):
    frame = Image.new("RGB", (1280, 400))
    for column, name in enumerate((first_view, second_view)):
        rgb = np.asarray(cameras[name].render()[0])
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb*255 if rgb.max() <= 1 else rgb, 0, 255).astype(np.uint8)
        panel = Image.fromarray(rgb[..., :3]).resize((640, 400), Image.Resampling.LANCZOS)
        frame.paste(panel, (column*640, 0))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 1279, 45), fill=(10, 18, 25))
    draw.text((12, 4), LABEL, fill=(236, 237, 233), font=font(15))
    draw.text((12, 24), f"t = {timestamp:05.2f} s  |  {stage}", fill=(202, 215, 220), font=font(14))
    draw.text((650, 24), f"{first_view} + {second_view}  |  recorded states only", fill=(202, 215, 220), font=font(14))
    draw.line((640, 46, 640, 399), fill=(120, 130, 135), width=2)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed output directory, or its name under outputs/")
    parser.add_argument("--output", help="Output directory; defaults to <run>/replay")
    parser.add_argument("--second-view", choices=("back", "detail"), default="back")
    parser.add_argument("--close-up", action="store_true", help="Show the garment closely from front and back")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    run_dir = resolve_run(args.run)
    config, positions, joints, times, rest, faces, stages, frame_dt, stage_source = load_recording(run_dir)
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "replay"
    output.mkdir(parents=True, exist_ok=True)

    from simulate import make_scene, arr
    scene_args = SimpleNamespace(backend=args.backend, viewer=False, no_render=False)
    scene, cloth, robots, cameras, _ = make_scene(config, scene_args)
    first_view="detail" if args.close_up else "perspective"
    if args.close_up:
        args.second_view="back"
        cameras['back'].set_pose(pos=(.12,.86,1.29),lookat=(0,0,.99))
    current_rest = arr(cloth.get_particles_pos())
    current_faces = np.asarray(cloth._mesh.faces)
    if current_rest.shape != rest.shape or cloth.n_particles != len(rest):
        raise ValueError("Current cloth particle count differs from the recorded topology")
    if not np.allclose(current_rest, rest, atol=1e-6, rtol=0):
        raise ValueError("Current rest vertices differ in coordinates or order from the recording")
    if not np.array_equal(current_faces, faces):
        raise ValueError("Current cloth face indices differ from the recorded topology")
    for k, robot in enumerate(robots):
        if arr(robot.get_qpos()).shape != joints[0, k].shape:
            raise ValueError(f"Robot {k} joint coordinates differ from the recording")
    print("TOPOLOGY_VERIFIED; replay only, no dynamics steps", flush=True)

    import imageio.v2 as imageio
    selected = contact_indices(stages)
    captured = {}
    video_path = output / "overview.mp4"
    with imageio.get_writer(video_path, fps=1/frame_dt, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p") as writer:
        for index, timestamp in enumerate(times):
            # Deliberate saved-state restoration for playback, never presented
            # as cloth control or as the physics used to generate the trajectory.
            cloth.set_particles_pos(positions[index])
            for k, robot in enumerate(robots):
                robot.set_qpos(joints[index, k])
            scene.visualizer.update(force=True)
            frame = compose_frame(cameras, first_view, args.second_view, float(timestamp), stages[index])
            writer.append_data(np.asarray(frame))
            if index in selected:
                captured[index] = frame.copy()
            if index % 25 == 0 or index == len(times)-1:
                print(f"REPLAY_FRAME {index+1}/{len(times)} t={timestamp:.2f} stage={stages[index]}", flush=True)

    rows = (len(selected)+1)//2
    sheet = Image.new("RGB", (1280, 54+rows*232), (16, 24, 31))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), LABEL, fill="white", font=font(19))
    draw.text((12, 32), "Sampled saved states; this contact sheet does not establish task success.",
              fill=(185, 200, 207), font=font(13))
    for item, index in enumerate(selected):
        x, y = (item % 2)*640, 54+(item//2)*232
        # Trim the video annotation and provide a readable panel-specific label.
        panel = captured[index].crop((0, 46, 1280, 400)).resize((640, 177), Image.Resampling.LANCZOS)
        sheet.paste(panel, (x, y+34))
        draw.text((x+12, y+8), f"{times[index]:05.2f} s  |  {stages[index]}", fill="white", font=font(16))
    sheet.save(output / "stage_contact_sheet.png")
    metadata = {
        "mode": "saved Genesis state playback, not new dynamics",
        "source_run": str(run_dir), "video": str(video_path), "label": LABEL,
        "frames": len(times), "saved_state_interval_s": frame_dt, "video_fps": 1/frame_dt,
        "first_saved_time_s": float(times[0]), "last_saved_time_s": float(times[-1]),
        "simulation_steps_executed": 0, "interpolated_states": 0,
        "cloth_topology_verification": "particle count, rest coordinates/order and exact face indices matched",
        "stage_labels_source": stage_source, "views": [first_view, args.second_view],
        "close_up": args.close_up,
        "contact_sheet_samples": [{"frame": i, "time": float(times[i]), "stage": stages[i]} for i in selected],
    }
    (output / "replay_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"REPLAY_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
