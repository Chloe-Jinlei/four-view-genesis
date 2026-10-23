"""Audit saved actual robot poses in a separate native rigid collision scene.

Example: python planning/check_actual_collisions.py --run wrap_v13
The run must be complete. This does not step or change the simulation. It
checks every saved pose, with native self-collision filtering enabled, against
the run's bear collision geometry, tabletop, and the other robot. The floor,
table legs, and fixed robot mounting plates are outside this audit's scope.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', default='wrap_v13')
args = parser.parse_args()
out = runtime.ROOT / 'outputs' / args.run
if not (out / 'summary.json').is_file():
    raise RuntimeError(f'{args.run} is not complete: summary.json is missing.')
cfg = json.loads((out / 'config.json').read_text())
manifest_path = runtime.ROOT / cfg.get('asset_manifest', 'assets/manifest.json')
manifest = json.loads(manifest_path.read_text())

import numpy as np
import genesis as gs
from genesis.utils.misc import qd_to_numpy

gs.init(backend=gs.cpu, logging_level='error', seed=cfg['seed'])
scene = gs.Scene(show_viewer=False,
                 rigid_options=gs.options.RigidOptions(enable_self_collision=True))
names = {}
material = gs.materials.Rigid(needs_coup=False)
for part in manifest['bear']['parts']:
    collides = (part['collision'] or part['name'] == 'belly_patch') and not cfg.get('bear_union_collision', False)
    entity = scene.add_entity(gs.morphs.Mesh(
        file=str(runtime.ROOT / 'assets' / part['mesh']), pos=part['position'],
        quat=part['quaternion_wxyz'], fixed=True, collision=collides,
        convexify=True, decimate=False), material)
    names[entity.idx] = part['name']
if cfg.get('bear_union_collision', False):
    entity = scene.add_entity(gs.morphs.Mesh(
        file=str(runtime.ROOT / 'assets/bear_collision_union.obj'), fixed=True,
        visualization=False, convexify=False, decimate=False), material)
    names[entity.idx] = 'bear_collision_union'
# These fixed dimensions and mounting height match make_scene in simulate.py.
table = scene.add_entity(gs.morphs.Box(
    pos=(0, 0, .6975), size=(1.35, 1., .045), fixed=True), material)
names[table.idx] = 'table'
robots = []
for side in [-1, 1]:
    robot = scene.add_entity(gs.morphs.MJCF(
        file='xml/franka_emika_panda/panda.xml',
        pos=(side * cfg['robot_base_x'], cfg['robot_base_y'], .735),
        euler=(0, 0, 0 if side < 0 else 180)), material)
    robots.append(robot)
    names[robot.idx] = 'robot_left' if side < 0 else 'robot_right'
scene.build()

trajectory = np.load(out / 'trajectory.npz')
actual_qpos, times = trajectory['joints'], trajectory['times']
metrics = json.loads((out / 'metrics.json').read_text())
metric_by_time = {round(float(m['time']), 6): m for m in metrics}
robot_ids = {r.idx for r in robots}
worst, detailed, joint_errors = {}, [], []
for t, qpair in zip(times, actual_qpos):
    for robot, q in zip(robots, qpair):
        robot.set_qpos(q)
    pairs = scene.rigid_solver.detect_collision()
    depths = qd_to_numpy(scene.rigid_solver.collider.collider_state.contact_data.penetration)
    contacts = {}
    for index, (a, b) in enumerate(pairs):
        ga, gb = scene.rigid_solver.geoms[a], scene.rigid_solver.geoms[b]
        if not ({ga.entity.idx, gb.entity.idx} & robot_ids):
            continue
        key = ' / '.join(sorted([
            names[ga.entity.idx] + ':' + ga.link.name,
            names[gb.entity.idx] + ':' + gb.link.name]))
        depth = float(depths[index, 0])
        contacts[key] = max(contacts.get(key, 0), depth)
        if key not in worst or depth > worst[key]['depth_m']:
            worst[key] = {'time': float(t), 'depth_m': depth}
    if contacts:
        detailed.append({'time': float(t), 'contacts': contacts})
    metric = metric_by_time.get(round(float(t), 6))
    if metric and all('commanded_qpos' in x for x in metric['robot_tracking']):
        targets = np.array([x['commanded_qpos'] for x in metric['robot_tracking']])
        joint_errors.append(np.max(np.abs(targets[:, :7] - qpair[:, :7]), axis=1))

result = {
    'run': args.run,
    'state_kind': 'ACTUAL saved simulated joint positions, NOT commanded PD targets',
    'samples': len(times), 'time_range_s': [float(times[0]), float(times[-1])],
    'sample_period_s': float(np.median(np.diff(times))),
    'self_collision_enabled_in_diagnostic': True,
    'physics_self_collision_setting': False,
    'includes': 'All robot native collision geometry vs own nonadjacent links under native pair filters, other robot, run bear geometry, and tabletop',
    'excluded': 'Floor, table legs, and fixed mounting plates',
    'collision_frames': len(detailed),
    'first_collision': detailed[0]['time'] if detailed else None,
    'maximum_penetration_m': max((v['depth_m'] for v in worst.values()), default=0.),
    'worst_contacts': worst, 'contact_samples': detailed,
    'max_command_minus_actual_joint_error_rad_left_right': np.max(joint_errors, axis=0).tolist() if joint_errors else None,
    'geometry_manifest': str(manifest_path.relative_to(runtime.ROOT)),
    'geometry_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'actual_joint_array_sha256': hashlib.sha256(np.ascontiguousarray(actual_qpos).tobytes()).hexdigest(),
    'limitations': 'Discrete saved samples only; does not certify continuous swept clearance or intervening transient contacts. No physics/config changes.',
}
(out / 'actual_joint_collision_report.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'contact_samples'}, indent=2))
