"""Generate and fully screen the outward45 point-pinch backup trajectory.

Uses native Genesis IK/FK and rigid collision geometry in a separate scene.
Run with the installed Genesis Python; default output is
planning/outward_backup_reproduced.*, leaving the checked original unchanged.
Use --output-prefix /tmp/outward_reproduced for an isolated reproduction.

Only joint command artifacts are written; simulation state is never replayed
by this planner. It reads image-reconstruction assets and the generated v6
closure seed, not any unknown original simulation.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-config',type=Path,default=runtime.ROOT/'planning/outward_backup_input.json')
parser.add_argument('--output-prefix',type=Path,default=runtime.ROOT/'planning/outward_backup_reproduced')
args=parser.parse_args()
output_prefix=args.output_prefix.resolve();output_prefix.parent.mkdir(parents=True,exist_ok=True)
import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation
from genesis.utils.misc import qd_to_numpy

def arr(v):return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
cfg=json.loads(args.input_config.read_text())
cfg['asset_manifest']='assets/manifest_connected.json'
manifest=json.load(open(runtime.ROOT/cfg['asset_manifest']))
gs.init(backend=gs.cpu,logging_level='error',seed=7)
s=gs.Scene(show_viewer=False,rigid_options=gs.options.RigidOptions(enable_self_collision=False))
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

from simulate import target_at
cfg['tool_orientation']='outward_wrist'
for key in cfg['trajectory']:
 key['tilt']=0. if key['time']<=2 else 45.
 if key['time'] in [13.,14.5]:key['right_y_offset']=.055
times=np.arange(0,17.001,.02);tclose=650
seedq=np.asarray(json.loads((runtime.ROOT/'planning/outward_backup_seed.json').read_text())['qpos_left_right'],dtype=np.float32)
paths={};errors={};angles={}
for k,(side,r) in enumerate(zip([-1,1],robots)):
 for roll in ([180] if k==0 else [0]):
  h=r.get_link('hand');qs=np.zeros((len(times),9),dtype=np.float32);err=np.zeros(len(times));angerr=np.zeros(len(times))
  def desired(t):
   p,o,_=target_at(cfg,float(t),side)
   if roll:o=(Rotation.from_quat(o,scalar_first=True)*Rotation.from_euler('z',roll,degrees=True)).as_quat(scalar_first=True)
   return p,o
  p,o=desired(13.);qclose=arr(r.inverse_kinematics(h,pos=p,quat=o,local_point=cfg['robot_tip_offset'],init_qpos=seedq[k],rot_mask=[True,True,True],max_samples=20,max_solver_iters=200,max_step_size=.1,damping=.05)).copy();qclose[-2:]=.004
  for indices in [range(tclose,-1,-1),range(tclose+1,len(times))]:
   prev=qclose.copy();r.set_qpos(prev)
   for i in indices:
    t=times[i];p,o=desired(t)
    if i==tclose:q=qclose.copy()
    else:
     q=arr(r.inverse_kinematics(h,pos=p,quat=o,local_point=cfg['robot_tip_offset'],init_qpos=prev,rot_mask=[True,True,True],max_samples=1,max_solver_iters=100,max_step_size=.03,damping=.1)).copy()
     q[:7]=prev[:7]+np.clip(q[:7]-prev[:7],-.04,.04)
    q[-2:]=.004 if 1.7<=t<14.5 else .04;r.set_qpos(q);prev=q;qs[i]=q
    rot=Rotation.from_quat(arr(h.get_quat()),scalar_first=True);tip=arr(h.get_pos())+rot.apply(cfg['robot_tip_offset']);err[i]=np.linalg.norm(tip-p);angerr[i]=np.rad2deg((rot.inv()*Rotation.from_quat(o,scalar_first=True)).magnitude())
  # Before grasp and after release, only the fingertip position matters.
  # Rebuild these segments from the held endpoint, retaining the branch/rate bound.
  for indices,start in [(range(99,-1,-1),100),(range(726,len(times)),725)]:
   prev=qs[start].copy();r.set_qpos(prev)
   for i in indices:
    t=times[i];p,o=desired(t)
    q=arr(r.inverse_kinematics(h,pos=p,quat=None,local_point=cfg['robot_tip_offset'],init_qpos=prev,max_samples=1,max_solver_iters=100,max_step_size=.03,damping=.1)).copy()
    q[:7]=prev[:7]+np.clip(q[:7]-prev[:7],-.04,.04);q[-2:]=.004 if 1.7<=t<14.5 else .04;r.set_qpos(q);prev=q;qs[i]=q
    rot=Rotation.from_quat(arr(h.get_quat()),scalar_first=True);tip=arr(h.get_pos())+rot.apply(cfg['robot_tip_offset']);err[i]=np.linalg.norm(tip-p);angerr[i]=np.rad2deg((rot.inv()*Rotation.from_quat(o,scalar_first=True)).magnitude())
  paths[(k,roll)]=qs;errors[(k,roll)]=err;angles[(k,roll)]=angerr
  held=(times>=2)&(times<14.5)
  print('BACKWARD45',side,'roll',roll,'held_max_tip',float(err[held].max()),'held_max_angle',float(angerr[held].max()),'all_max_tip',float(err.max()),'t_worst',float(times[np.argmax(err)]),flush=True)
  for tt in [2,4,6,7,8,9,10,11,12,13,14.5,17]:
   ii=int(round(tt/.02));print('BACKWARD_SAMPLE',side,roll,tt,'tip',float(err[ii]),'angle',float(angerr[ii]),flush=True)

left_roll,right_roll=180,0
commands=np.stack([paths[(0,left_roll)],paths[(1,right_roll)]],axis=1)
worst={};first=None;colliding=0;collision_samples=[]
for i,t in enumerate(times):
 robots[0].set_qpos(commands[i,0]);robots[1].set_qpos(commands[i,1]);pairs=s.rigid_solver.detect_collision();depth=qd_to_numpy(s.rigid_solver.collider.collider_state.contact_data.penetration)
 hit=False
 for j,(a,b) in enumerate(pairs):
  ga,gb=s.rigid_solver.geoms[a],s.rigid_solver.geoms[b];na,nb=names[ga.entity.idx]+':'+ga.link.name,names[gb.entity.idx]+':'+gb.link.name
  if 'robot_' not in na and 'robot_' not in nb:continue
  hit=True;key=' / '.join(sorted([na,nb]));d=float(depth[j,0])
  if key not in worst or d>worst[key]['depth_m']:worst[key]={'time':float(t),'depth_m':d}
 if hit:
  colliding+=1;collision_samples.append(float(t))
  if first is None:first=float(t)
held=(times>=2)&(times<14.5);wrap=(times>=6)&(times<14.5)
cfg['local_tool_roll_degrees']=[180,0];cfg['joint_trajectory']=str(output_prefix.with_suffix('.npy').relative_to(runtime.ROOT)) if output_prefix.is_relative_to(runtime.ROOT) else str(output_prefix.with_suffix('.npy'));cfg['joint_trajectory_dt']=.02
result={'left_roll':180,'right_roll':0,'held_max_tip_error_m':[float(errors[(0,180)][held].max()),float(errors[(1,0)][held].max())],'wrap_max_tip_error_m':[float(errors[(0,180)][wrap].max()),float(errors[(1,0)][wrap].max())],'all_motion_max_tip_error_m':[float(errors[(0,180)].max()),float(errors[(1,0)].max())],'grasp_t1_9_tip_error_m':[float(errors[(0,180)][95]),float(errors[(1,0)][95])],'maximum_joint_step_rad':float(np.max(np.abs(np.diff(commands[:,:,:7],axis=0)))),'collision_frames':colliding,'first_collision':first,'worst_contacts':worst,'asset_manifest':cfg['asset_manifest'],'initial_qpos':commands[0].tolist(),'period_seconds':.02,'times_seconds':times.tolist(),'orientation_policy':'Full outward45 with local rolls[180,0] during2–14.5s. Position-only approach0–2 and retreat14.5–17s; no cloth attached except1.9–14.5.','validation_scope':'All robot rigid collision geometry vs connected bear and other robot at every20ms. Native self-collision flag matches simulation(False). No dynamics or swept collision guarantee.'}
print('BACKUP_RESULT',json.dumps({k:v for k,v in result.items() if k!='times_seconds'}),flush=True)
stem=output_prefix;np.save(stem.with_suffix('.npy'),commands);stem.with_suffix('.json').write_text(json.dumps(result,indent=2));stem.with_name(stem.name+'_config').with_suffix('.json').write_text(json.dumps(cfg,indent=2));np.savez_compressed(stem.with_name(stem.name+'_checks').with_suffix('.npz'),times=times,qpos=commands,position_errors_m=np.stack([errors[(0,180)],errors[(1,0)]],axis=1),orientation_errors_degrees=np.stack([angles[(0,180)],angles[(1,0)]],axis=1))
