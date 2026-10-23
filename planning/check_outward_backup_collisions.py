"""Screen ALL robot collision geometry, including native self-collisions.

Separate diagnostic scene only. Checks851 saved poses at20ms; does not step
physics or alter the running simulation. Includes connected bear and table.
"""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
import numpy as np
import genesis as gs
from genesis.utils.misc import qd_to_numpy

cfg=json.load(open(runtime.ROOT/'planning/outward_backup_config.json'))
manifest=json.load(open(runtime.ROOT/cfg['asset_manifest']))
gs.init(backend=gs.cpu,logging_level='error',seed=cfg['seed'])
s=gs.Scene(show_viewer=False,rigid_options=gs.options.RigidOptions(enable_self_collision=True))
names={};bears=[]
for part in manifest['bear']['parts']:
 collides=part['collision'] or part['name']=='belly_patch'
 e=s.add_entity(gs.morphs.Mesh(file=str(runtime.ROOT/'assets'/part['mesh']),pos=part['position'],quat=part['quaternion_wxyz'],fixed=True,collision=collides,convexify=True,decimate=False),gs.materials.Rigid(needs_coup=False))
 names[e.idx]=part['name'];bears.append(e)
table=s.add_entity(gs.morphs.Box(pos=(0,0,.6975),size=(1.35,1.,.045),fixed=True),gs.materials.Rigid(needs_coup=False));names[table.idx]='table'
robots=[]
for side in [-1,1]:
 r=s.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml',pos=(side*cfg['robot_base_x'],cfg['robot_base_y'],.735),euler=(0,0,0 if side<0 else 180)),gs.materials.Rigid(needs_coup=False));robots.append(r);names[r.idx]='robot_left' if side<0 else 'robot_right'
s.build();print('BUILT',flush=True)


commands=np.load(runtime.ROOT/'planning/outward_backup_joint_path.npy');worst={};frames=[]
for i,qpair in enumerate(commands):
 for r,q in zip(robots,qpair):r.set_qpos(q)
 pairs=s.rigid_solver.detect_collision();depth=qd_to_numpy(s.rigid_solver.collider.collider_state.contact_data.penetration);hit=False
 for j,(a,b) in enumerate(pairs):
  ga,gb=s.rigid_solver.geoms[a],s.rigid_solver.geoms[b]
  if not (ga.entity in robots or gb.entity in robots):continue
  key=' / '.join(sorted([names[ga.entity.idx]+':'+ga.link.name,names[gb.entity.idx]+':'+gb.link.name]));d=float(depth[j,0]);hit=True
  if key not in worst or d>worst[key]['depth_m']:worst[key]={'time':i*.02,'depth_m':d}
 if hit:frames.append(i*.02)
result={'self_collision_enabled':True,'collision_frames':len(frames),'first_collision':frames[0] if frames else None,'worst_contacts':worst,'samples':len(commands),'period_s':.02,'state_kind':'Nominal joint commands, not actual simulated robot states','scope':'All robot native collision links, their own nonadjacent links under native filters, the other robot, connected bear, and tabletop. Discrete samples only; no swept or dynamic clearance guarantee.','config_file':'planning/outward_backup_config.json','joint_path_file':'planning/outward_backup_joint_path.npy','joint_path_sha256':hashlib.sha256((runtime.ROOT/'planning/outward_backup_joint_path.npy').read_bytes()).hexdigest(),'geometry_manifest':cfg['asset_manifest'],'geometry_manifest_sha256':hashlib.sha256((runtime.ROOT/cfg['asset_manifest']).read_bytes()).hexdigest()}
print('FULL_INCLUDING_SELF',json.dumps(result),flush=True)
open(runtime.ROOT/'planning/outward_backup_self_collision_report.json','w').write(json.dumps(result,indent=2))
