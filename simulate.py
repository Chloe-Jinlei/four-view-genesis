"""Image-only reconstruction in native Genesis 1.4.3.

No per-frame global cloth morphing: native PBD evolves the cloth. The only
external cloth constraints are small ideal gripper patches and contact seams.
"""
import runtime
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import genesis as gs

ROOT = runtime.ROOT


def arr(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def quat(tilt, yaw=0):
    rot=Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_euler('x',180+tilt,degrees=True)
    return rot.as_quat(scalar_first=True)


def target_at(config, t, side):
    keys = config['trajectory']
    i = min(np.searchsorted([k['time'] for k in keys], t, side='right')-1, len(keys)-2)
    i = max(0, i)
    a,b = keys[i:i+2]
    u = np.clip((t-a['time'])/(b['time']-a['time']),0,1)
    u = u*u*(3-2*u)
    vals = {k: a[k] + (b[k]-a[k])*u for k in ('half_span','y','z','tilt')}
    yaw=a.get('yaw',0)+(b.get('yaw',0)-a.get('yaw',0))*u
    # Diagonal material grasp points keep the wrists apart at the back seam.
    # Their 12 cm material separation rotates from world Y into world Z.
    tilt=np.deg2rad(vals['tilt'])
    offset=np.zeros(3)
    y_key='left_y_offset' if side<0 else 'right_y_offset'
    offset[1]=a.get(y_key,0)+(b.get(y_key,0)-a.get(y_key,0))*u
    lowering=config.get('sleeve_lowering',{})
    if lowering:
        v=np.clip((t-lowering['start'])/(lowering['end']-lowering['start']),0,1)
        v=v*v*(3-2*v)
        r=np.clip((t-14.5)/2.5,0,1)
        offset[2]=-lowering['distance']*v*(1-r*r*(3-2*r))
    closure_lowering=config.get('closure_lowering',{})
    if closure_lowering:
        v=np.clip((t-closure_lowering['start'])/(closure_lowering['end']-closure_lowering['start']),0,1)
        v=v*v*(3-2*v)
        r=np.clip((t-14.5)/2.5,0,1)
        offset[2]-=closure_lowering['distance']*v*(1-r*r*(3-2*r))
    if config.get('stagger_grasps',False) and side>0:
        grasp_offset=Rotation.from_euler('z',side*yaw,degrees=True).apply([0,-.12*np.cos(tilt),-.12*np.sin(tilt)])
        # The grasp offset is introduced while approaching the initially flat garment.
        offset+=grasp_offset*min(t/1.5,1.0)
    yaw_sign=-side if config.get('reverse_yaw_paths',False) else side
    if config.get('tool_orientation')=='outward_wrist':
        rotation=Rotation.from_euler('y',side*vals['tilt'],degrees=True)*Rotation.from_euler('x',180,degrees=True)
        local_roll=config.get('local_tool_roll_degrees',[0,0])[0 if side<0 else 1]
        rotation=rotation*Rotation.from_euler('z',local_roll,degrees=True)
        orientation=rotation.as_quat(scalar_first=True)
    else:
        orientation=quat(vals['tilt'],yaw_sign*yaw)
    return np.array([side*vals['half_span'],vals['y'],vals['z']])+offset,orientation,b['stage']


def make_scene(config, args):
    gs.init(backend=gs.cpu if args.backend=='cpu' else gs.gpu, seed=config['seed'], logging_level='warning')
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config['dt'],substeps=config['substeps']),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        pbd_options=gs.options.PBDOptions(particle_size=config['particle_size'],
            max_stretch_solver_iterations=config.get('stretch_iterations',16),max_bending_solver_iterations=3,
            lower_bound=(-.9,-.8,.65),upper_bound=(.9,.8,1.65)),
        vis_options=gs.options.VisOptions(background_color=(.025,.045,.06),
            ambient_light=(.35,.35,.35),shadow=True,plane_reflection=False),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.65,-2.1,1.8),camera_lookat=(0,0,1.0),camera_fov=43),
        show_viewer=args.viewer,
    )
    solid=gs.materials.Rigid(needs_coup=True,coup_softness=.001,sdf_cell_size=.004)
    decoration=gs.materials.Rigid(needs_coup=False)
    scene.add_entity(gs.morphs.Box(pos=(0,0,-.03),size=(200,200,.05),fixed=True),
        material=decoration,surface=gs.surfaces.Rough(color=(.21,.23,.25,1)))
    scene.add_entity(gs.morphs.Box(pos=(0,0,.72-.0225),size=(1.35,1,.045),fixed=True),
        material=solid,surface=gs.surfaces.Rough(color=(.46,.30,.18,1)))
    for x in [-.58,.58]:
        for y in [-.41,.41]:
            scene.add_entity(gs.morphs.Box(pos=(x,y,.34),size=(.048,.048,.68),fixed=True),
                material=decoration,surface=gs.surfaces.Rough(color=(.10,.14,.15,1)))
    manifest=json.loads((ROOT/config.get('asset_manifest','assets/manifest.json')).read_text())
    bear={}
    for p in manifest['bear']['parts']:
        collides=(p['collision'] or p['name']=='belly_patch') and not config.get('bear_union_collision',False)
        bear[p['name']]=scene.add_entity(gs.morphs.Mesh(file=str(ROOT/'assets'/p['mesh']),
            pos=p['position'],quat=p['quaternion_wxyz'],fixed=True,collision=collides,
            convexify=True,decimate=False), material=solid if collides else decoration,
            surface=gs.surfaces.Rough(color=p['color_rgba']))
    if config.get('bear_union_collision',False):
        scene.add_entity(gs.morphs.Mesh(file=str(ROOT/'assets/bear_collision_union.obj'),
            fixed=True,visualization=False,convexify=False,decimate=False),material=solid)
    robots=[]
    for side in [-1,1]:
        base=(side*config['robot_base_x'],config['robot_base_y'],.735)
        scene.add_entity(gs.morphs.Box(pos=(base[0],base[1],.7275),size=(.19,.19,.015),fixed=True),
            material=decoration,surface=gs.surfaces.Rough(color=(.12,.16,.18,1)))
        robot=scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml',
            pos=base,euler=(0,0,0 if side<0 else 180)),
            material=gs.materials.Rigid(gravity_compensation=1.0,needs_coup=False))
        robots.append(robot)
    cloth=scene.add_entity(gs.morphs.Mesh(file=str(ROOT/'assets/cloth.obj'),
        pos=config['cloth_initial_center'],decimate=False,convexify=False),
        material=gs.materials.PBD.Cloth(**config['cloth_material']),
        surface=gs.surfaces.Rough(double_sided=True))
    cameras={}
    poses={
        'perspective':((1.65,-2.1,1.85),(0,0,1.0)),
        'top':((0,-.001,2.85),(0,0,.72)),
        'side':((-2.0,-.3,1.50),(0,0,1.02)),
        'back':((0,2.4,1.65),(0,0,1.02)),
        'detail':((.58,-.83,1.38),(0,0,.98)),
    }
    if not args.no_render:
        for name,(pos,lookat) in poses.items():
            cameras[name]=scene.add_camera(res=(960,600),pos=pos,lookat=lookat,fov=42 if name!='top' else 35,GUI=False)
    print('BUILDING_NATIVE_SCENE',flush=True)
    scene.build()
    scene._reconstruction_bear_entities=list(bear.values())
    print('BUILT',cloth.n_particles,'cloth particles',flush=True)
    planned=(np.load(ROOT/config['joint_trajectory']) if config.get('joint_trajectory') else None)
    for robot_index,(side,robot) in enumerate(zip([-1,1],robots)):
        q0=np.array([0,-.45,0,-2.0,0,1.6,.785,.04,.04])
        robot.set_qpos(q0)
        robot.set_dofs_kp([1600,1600,1400,1400,700,700,500,160,160])
        robot.set_dofs_kv([120,120,100,100,55,55,40,12,12])
        robot.set_dofs_force_range([-87]*4+[-12]*3+[-40]*2,[87]*4+[12]*3+[40]*2)
        target,q,_=target_at(config,0,side)
        if planned is not None:
            solved=planned[0,robot_index].copy()
        else:
            solved=robot.inverse_kinematics(robot.get_link('hand'),pos=target,quat=q,
                local_point=config['robot_tip_offset'],max_samples=20,max_solver_iters=60,
                rot_mask=config.get('ik_rot_mask',[True,True,True]))
        solved[-2:]=.04
        robot.set_qpos(solved)
        robot.control_dofs_position(solved)
    return scene,cloth,robots,cameras,manifest


def snap(cameras,out,prefix):
    for name,camera in cameras.items():
        rgb=camera.render()[0]
        Image.fromarray(rgb).save(out/f'{prefix}_{name}.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--output',default='run_01')
    parser.add_argument('--duration',type=float,default=17.)
    parser.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--render-every',type=int,default=10)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--initial-only',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    out=ROOT/'outputs'/args.output
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    asset_manifest=config.get('asset_manifest','assets/manifest.json')
    asset_data=json.loads((ROOT/asset_manifest).read_text())
    inputs=['simulate.py','runtime.py','velcro.py','contact_friction.py','validation.py',
        'assess_run.py',asset_manifest,'assets/cloth.obj','assets/cloth.mtl']
    inputs.extend(str(Path('assets')/p['mesh']) for p in asset_data['bear']['parts'])
    if config.get('joint_trajectory'):
        inputs.append(config['joint_trajectory'])
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in sorted(set(inputs)) if (ROOT/name).is_file()}
    (out/'source_manifest.json').write_text(json.dumps({'sha256':hashes,
        'config_sha256':hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),
        'engine':gs.__version__},indent=2))
    scene,cloth,robots,cameras,manifest=make_scene(config,args)
    friction=None
    if config.get('bear_contact_friction'):
        from contact_friction import FixedBodyContactFriction
        friction=FixedBodyContactFriction(cloth,scene._reconstruction_bear_entities,
            **config['bear_contact_friction'])
    rest=arr(cloth.get_particles_pos()).copy()
    faces=np.asarray(cloth._mesh.faces).copy()
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    rest_lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    np.savez_compressed(out/'topology.npz',rest=rest,faces=faces)
    snap(cameras,out,'initial')
    if args.initial_only:
        print('INITIAL_SCENE_COMPLETE',flush=True)
        return
    from velcro import VelcroSeam,match_boundary_pairs,boundary_strips
    capacity=config.get('velcro_max_pairs',16)
    pairs=match_boundary_pairs(rest,width=.028 if capacity>16 else .012,order_axis=1,max_pairs=capacity)
    left_strip,right_strip=boundary_strips(rest,width=.028)
    seam=VelcroSeam(cloth,pairs,enabled=False,attach_distance=.020,max_attach_speed=.12,break_extension=.045,
        contact_mode=config.get('velcro_contact_mode','fixed'),left_strip=left_strip,right_strip=right_strip)
    grips=[]
    for side in [-1,1]:
        point=np.array([side*config.get('grip_local_uv',[.33,.06])[0],
            -.38 if side>0 and config.get('stagger_grasps',False) else -.26,.731])
        ids=np.flatnonzero((abs(rest[:,0]-point[0])<.014)&(abs(rest[:,1]-point[1])<.014))
        # A point pinch constrains translation while allowing the material to turn.
        # This is an explicitly idealized, zero-torque grasp, not cloth morphing.
        count=config.get('grasp_particle_count')
        if count is not None:
            ids=ids[np.argsort(np.linalg.norm(rest[ids]-point,axis=1))[:int(count)]]
            if len(ids)==0:
                raise ValueError('No particles available for the configured grasp')
        grips.append(ids)
    active_grips=[False,False]
    history=[]
    positions=[]
    joints=[]
    times=[]
    framewriter=None
    if args.video and cameras:
        import imageio.v2 as imageio
        framewriter=imageio.get_writer(out/'simulation.mp4',fps=1/(config['dt']*args.render_every),codec='libx264',quality=8,macro_block_size=1)
    start=time.monotonic()
    laststage=None
    aborted_reason=None
    prev_q=[arr(r.get_qpos()).copy() for r in robots]
    last_ik_error=[None,None]
    planned=(np.load(ROOT/config['joint_trajectory']) if config.get('joint_trajectory') else None)
    for step in range(int(args.duration/config['dt'])+1):
        t=step*config['dt']
        for k,(side,robot) in enumerate(zip([-1,1],robots)):
            target,q,stage=target_at(config,t,side)
            if step%2==0:
                fine_ik=t>=config.get('fine_ik_start',float('inf'))
                ik_options=({'max_samples':1,'max_solver_iters':80,'max_step_size':.02,'damping':.1}
                    if fine_ik else {'max_samples':2,'max_solver_iters':25})
                if planned is not None:
                    index=min(int(round(t/config.get('joint_trajectory_dt',.02))),len(planned)-1)
                    solved=gs.tensor(planned[index,k].copy())
                else:
                    solved,error=robot.inverse_kinematics(robot.get_link('hand'),pos=target,quat=q,
                        local_point=config['robot_tip_offset'],init_qpos=prev_q[k],
                        **ik_options,return_error=True,
                        rot_mask=config.get('ik_rot_mask',[True,True,True]))
                    last_ik_error[k]=arr(error).tolist()
                # Keep IK branch changes from turning into large servo commands.
                # At 50 Hz this limits each joint target to 2 radians/second.
                limited=prev_q[k][:7]+np.clip(arr(solved)[:7]-prev_q[k][:7],-.04,.04)
                solved[:7]=gs.tensor(limited)
                solved[-2:]=.004 if 1.7<=t<14.5 else .04
                prev_q[k]=arr(solved).copy()
                robot.control_dofs_position(solved)
            if not active_grips[k] and 1.9<=t<2.2:
                hand=robot.get_link('hand')
                hp=arr(hand.get_pos())
                hq=arr(hand.get_quat())
                tip=hp+Rotation.from_quat(hq,scalar_first=True).apply(config['robot_tip_offset'])
                distance=float(np.linalg.norm(arr(cloth.get_particles_pos())[grips[k]].mean(0)-tip))
                if distance < .03:
                    cloth.fix_particles_to_link(hand.idx,grips[k])
                    active_grips[k]=True
                    print('GRASP',k,'distance',distance,'vertices',len(grips[k]),flush=True)
            if active_grips[k] and t>=14.5:
                cloth.release_particle(grips[k])
                active_grips[k]=False
        if t>=11.0:
            seam.enable()
        seam.before_step()
        scene.step()
        if stage!=laststage:
            print('STAGE',stage,'t',round(t,2),'elapsed',round(time.monotonic()-start,1),flush=True)
            laststage=stage
            if cameras:
                Image.fromarray(cameras['detail'].render()[0]).save(out/f'stage_{step:04d}_{stage}.png')
        if step%10==0:
            p=arr(cloth.get_particles_pos()).copy()
            if not np.isfinite(p).all():
                raise RuntimeError('Nonfinite PBD state')
            metrics=seam.metrics()
            record={'time':t,'stage':stage,'cloth_min':p.min(0).tolist(),
                'cloth_max':p.max(0).tolist(),'grips_active':active_grips.copy(),
                'velcro':metrics}
            if friction is not None:
                record['bear_contact_friction']=friction.metrics()
            ratios=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)/rest_lengths
            record['p99_edge_length_ratio']=float(np.quantile(ratios,.99))
            record['max_particle_speed']=float(np.linalg.norm(arr(cloth.get_particles_vel()),axis=1).max())
            record['robot_tracking']=[]
            for k,(side,robot) in enumerate(zip([-1,1],robots)):
                hand=robot.get_link('hand')
                tip=arr(hand.get_pos())+Rotation.from_quat(arr(hand.get_quat()),scalar_first=True).apply(config['robot_tip_offset'])
                goal,_,_=target_at(config,t,side)
                record['robot_tracking'].append({'side':side,'tip':tip.tolist(),'target':goal.tolist(),
                    'position_error_m':float(np.linalg.norm(tip-goal)),
                    'ik_error':last_ik_error[k],
                    'commanded_qpos':prev_q[k].tolist(),
                    'max_joint_speed_rad_s':float(np.abs(arr(robot.get_dofs_velocity())[:7]).max())})
            history.append(record)
            positions.append(p)
            joints.append(np.stack([arr(r.get_qpos()) for r in robots]))
            times.append(t)
            if step%100==0:
                (out/'progress.json').write_text(json.dumps(record,indent=2))
                np.savez_compressed(out/'latest_state.npz',positions=p,
                    joints=np.stack([arr(r.get_qpos()) for r in robots]),time=t)
                print('STEP',step,'zmin',round(float(p[:,2].min()),3),'seam',metrics['active_count'],flush=True)
            if record['p99_edge_length_ratio']>3:
                aborted_reason=f'Numerical strain guard triggered at t={t:.2f}s'
                print('ABORT',aborted_reason,flush=True)
                break
        if framewriter and step%args.render_every==0:
            rgb=cameras['perspective'].render()[0]
            im=Image.fromarray(rgb)
            d=ImageDraw.Draw(im)
            d.rectangle((0,0,960,36),fill=(10,18,25))
            d.text((14,10),f'Genesis native PBD  |  {stage}  |  t={t:05.2f}s  |  IMAGE-ONLY APPROXIMATION',fill='white')
            framewriter.append_data(np.asarray(im))
    if framewriter: framewriter.close()
    snap(cameras,out,'final')
    np.savez_compressed(out/'trajectory.npz',positions=np.stack(positions),joints=np.stack(joints),times=times)
    (out/'metrics.json').write_text(json.dumps(history,indent=2))
    summary={'engine':gs.__version__,'backend':args.backend,'duration_s':args.duration,
        'actual_duration_s':times[-1],'aborted_reason':aborted_reason,
        'wall_seconds':time.monotonic()-start,'cloth_particles':len(rest),
        'velcro':seam.metrics(),'assumptions':'Image-only estimated geometry, rigid fixed teddy, ideal no-slip gripper patches, scripted IK + PD, custom XPBD seam. Not recovered original physics.'}
    try:
        from validation import analyze_cloth
        from assess_run import DEFAULT_GEOMETRY
        geometry=config.get('validation_geometry',DEFAULT_GEOMETRY)
        summary['geometry']=analyze_cloth(rest,faces,arr(cloth.get_particles_pos()),
            arms=geometry['arms'],torso_center=geometry['torso_center'],torso_radii=geometry['torso_radii'])
    except Exception as e:
        summary['validation_error']=repr(e)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__': main()
