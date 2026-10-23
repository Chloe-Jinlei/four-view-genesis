#!/usr/bin/env python3
"""Add explicit visible shoulder joins to the isolated fitted-bear hypothesis.

All older assets are preserved. New brown ellipsoids overlap the arm roots and
torso; a numerical diagnostic checks analytic overlap and actual mesh clearance.
"""
from __future__ import annotations
import copy
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parent


def shape(part, margin=0.):
    center=np.array(part['position'],float)
    radii=np.array(part['radii'],float)
    rotation=Rotation.from_quat(part['quaternion_wxyz'],scalar_first=True).as_matrix()
    radii*=1-margin/radii.min()
    return center,rotation@np.diag(radii)


def minimum_normalized_radius(a,b,margin=0.):
    """Convex quadratic minimization on a unit ball, solved by its multiplier."""
    ca,A=shape(a,margin); cb,B=shape(b,margin)
    inverse=np.linalg.inv(B); Q=inverse.T@inverse; d=ca-cb
    H=A.T@Q@A; g=A.T@Q@d
    def u(lam): return -np.linalg.solve(H+lam*np.eye(3),g)
    if np.linalg.norm(u(0)) <= 1: coordinate=u(0)
    else:
        high=1.
        while np.linalg.norm(u(high))>1: high*=2
        coordinate=u(brentq(lambda lam:np.linalg.norm(u(lam))-1,0,high,xtol=1e-14))
    point=ca+A@coordinate
    return float(np.linalg.norm(inverse@(point-cb))),point


def analytic_gap(a,b):
    ca,A=shape(a);cb,B=shape(b); M=np.column_stack((A,-B));d=ca-cb
    objective=lambda x:float(np.dot(d+M@x,d+M@x))
    jac=lambda x:2*M.T@(d+M@x)
    constraints=[{'type':'ineq','fun':lambda x:1-np.dot(x[:3],x[:3]),
                  'jac':lambda x:np.r_[-2*x[:3],np.zeros(3)]},
                 {'type':'ineq','fun':lambda x:1-np.dot(x[3:],x[3:]),
                  'jac':lambda x:np.r_[np.zeros(3),-2*x[3:]]}]
    result=minimize(objective,np.zeros(6),jac=jac,constraints=constraints,method='SLSQP',
                    options={'ftol':1e-17,'maxiter':1000})
    value,point=minimum_normalized_radius(a,b)
    return {'minimum_euclidean_gap_m':float(np.sqrt(max(result.fun,0))),
            'optimizer_success':bool(result.success),
            'minimum_normalized_body_radius_in_arm':value,
            'analytic_overlap':bool(value<1),
            'nearest_on_arm':(ca+A@result.x[:3]).tolist(),
            'nearest_on_body':(cb+B@result.x[3:]).tolist()}


def world_mesh(part):
    mesh=trimesh.load_mesh(ROOT/part['mesh'],process=False)
    transform=np.eye(4)
    transform[:3,:3]=Rotation.from_quat(part['quaternion_wxyz'],scalar_first=True).as_matrix()
    transform[:3,3]=part['position']
    mesh.apply_transform(transform)
    return mesh


def clearance_lower_bound(point,part):
    c,A=shape(part)
    normalized=float(np.linalg.norm(np.linalg.solve(A,point-c)))
    return float(min(part['radii'])*(1-normalized)),normalized


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('fitted','original'),default='fitted')
    args=parser.parse_args()
    is_original=args.variant=='original'
    source_name='manifest.json' if is_original else 'manifest_fitted.json'
    output_name='manifest_original_connected.json' if is_original else 'manifest_connected.json'
    diagnostic_name='arm_connectivity_original_diagnostic.json' if is_original else 'arm_connectivity_diagnostic.json'
    folder_name='bear_parts_original_connected' if is_original else 'bear_parts_connected'
    preview_name='bear_original_connected.glb' if is_original else 'bear_connected.glb'
    source_path=ROOT/source_name;source_bytes=source_path.read_bytes()
    fitted=json.loads(source_bytes);connected=copy.deepcopy(fitted)
    folder=ROOT/folder_name;folder.mkdir(exist_ok=True)
    if is_original:
        connected['bear']['validation_arms']={
            'left':{'shoulder':[-.09,0,1.025],'tip':[-.15,-.008,.925]},
            'right':{'shoulder':[.09,0,1.025],'tip':[.15,-.008,.925]}}
    diagnostic={'method':'Convex analytic ellipsoid checks plus actual faceted-mesh interior distances.',
                'required_overlap_clearance_m':.005,'tested_analytic_inner_margin_m':.006,
                'old_geometry':{},'joins':{}}
    for filename in ('manifest.json','manifest_fitted.json'):
        manifest=json.loads((ROOT/filename).read_text());parts={p['name']:p for p in manifest['bear']['parts']}
        diagnostic['old_geometry'][filename]={}
        for side in ('left','right'):
            arm=parts[f'arm_{side}']
            pair={name:analytic_gap(arm,parts[name]) for name in ('torso','head')}
            diagnostic['old_geometry'][filename][side]=pair

    parts={p['name']:p for p in connected['bear']['parts']}
    for side,sign in (('left',-1),('right',1)):
        name=f'shoulder_{side}'
        part={'name':name,'mesh':f'{folder_name}/{name}.obj',
              'position':[sign*(.075 if is_original else .090),-.007,1.016],'quaternion_wxyz':[1,0,0,0],
              'radii':[.032 if is_original else .034,.028,.032], 'color_rgba':parts[f'arm_{side}']['color_rgba'],
              'collision':True,'fixed':True,'geometry':'ellipsoid'}
        mesh=trimesh.creation.icosphere(subdivisions=3,radius=1.)
        mesh.vertices*=np.array(part['radii'])
        mesh.visual.vertex_colors=np.tile(np.rint(np.array(part['color_rgba'])*255).astype(np.uint8),(len(mesh.vertices),1))
        mesh.export(ROOT/part['mesh']);connected['bear']['parts'].append(part)
        bridge_mesh=world_mesh(part)
        check={}
        for other_name in (f'arm_{side}','torso'):
            other=parts[other_name]
            normalized,witness=minimum_normalized_radius(other,part,margin=.006)
            bounds={p['name']:clearance_lower_bound(witness,p)[0] for p in (other,part)}
            actual={other_name:float(trimesh.proximity.signed_distance(world_mesh(other),witness[None,:])[0]),
                    name:float(trimesh.proximity.signed_distance(bridge_mesh,witness[None,:])[0])}
            assert normalized<1 and min(bounds.values())>=.006-1e-9
            assert min(actual.values())>=.005,actual
            check[other_name]={'witness_world':witness.tolist(),
                               'analytic_clearance_lower_bounds_m':bounds,
                               'actual_mesh_interior_clearance_m':actual,
                               'at_least_5mm_common_interior':True}
        shoulder=np.array(connected['bear']['validation_arms'][side]['shoulder'])
        arm=parts[f'arm_{side}'];c,A=shape(arm);pole=c+A[:,2]
        for label,point in (('nominal_shoulder_point',shoulder),('upper_arm_pole',pole)):
            bound,norm=clearance_lower_bound(point,part)
            actual=float(trimesh.proximity.signed_distance(bridge_mesh,point[None,:])[0])
            check[label]={'position':point.tolist(),'inside_bridge':bool(norm<1),
                          'analytic_clearance_lower_bound_m':bound,'mesh_clearance_m':actual}
            assert actual>.005
        diagnostic['joins'][side]=check
    connected['provenance']='Independent image-inferred bear with explicit shoulder joins correcting disconnected arm geometry; not original simulation ground truth.'
    connected['variant']={
        'name':'original_geometry_connected_shoulders' if is_original else 'fuller_torso_connected_shoulders','source_manifest':source_name,
        'source_manifest_sha256':hashlib.sha256(source_bytes).hexdigest(),'is_ground_truth':False,
        'changed_parts':[], 'added_parts':['shoulder_left','shoulder_right'],
        'reason':'Both prior arm ellipsoids were separated from the torso. Brown shoulder joins bury the upper caps and provide real collision continuity.',
        'diagnostic':diagnostic_name,
        'assumptions':['Shoulder shape and stiffness remain independently inferred.',
                       'Original and fitted manifests/meshes are preserved for their existing recordings.',
                       'The original collision-union mesh is not applicable to this variant.']}
    connected['bear']['modeling_assumptions'].append('Two visible fixed brown shoulder ellipsoids connect each arm root to the torso with verified >=5 mm interior margin.')
    scene=trimesh.Scene()
    for p in connected['bear']['parts']:
        mesh=world_mesh(p)
        mesh.visual.vertex_colors=np.tile(np.rint(np.array(p['color_rgba'])*255).astype(np.uint8),(len(mesh.vertices),1))
        scene.add_geometry(mesh,node_name=p['name'],geom_name=p['name'])
    (ROOT/preview_name).write_bytes(scene.export(file_type='glb'))
    connected['bear']['world_visual_glb']=preview_name
    (ROOT/output_name).write_text(json.dumps(connected,ensure_ascii=False,indent=2)+'\n')
    (ROOT/diagnostic_name).write_text(json.dumps(diagnostic,indent=2)+'\n')
    assert source_path.read_bytes()==source_bytes
    assert connected['bear']['parts'][:-2]==fitted['bear']['parts']
    assert connected['cloth']==fitted['cloth']
    print(json.dumps({'manifest':str(ROOT/output_name),'diagnostic':str(ROOT/diagnostic_name),
                      'added_parts':['shoulder_left','shoulder_right'],'source_manifest_preserved':True,
                      'original_parts_exactly_preserved':True},indent=2))


if __name__=='__main__':main()
