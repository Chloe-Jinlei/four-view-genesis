"""Native Genesis inclined-plane regression for the optional friction hook.

Run explicitly with the installed Genesis Python. This constructs a tiny scene
and tests the actual native collision hook with isolated analytical contacts;
it does not run or alter the bear dressing reconstruction.
"""
import runtime
import numpy as np
import quadrants as qd
import genesis as gs
from contact_friction import FixedBodyContactFriction, coulomb_tangent_correction

gs.init(backend=gs.cpu,logging_level='warning')
scene=gs.Scene(show_viewer=False,sim_options=gs.options.SimOptions(dt=.01,substeps=1,gravity=(0,0,0)),pbd_options=gs.options.PBDOptions(particle_size=.01,lower_bound=(-1,-1,-1),upper_bound=(1,1,1)))
plane=scene.add_entity(gs.morphs.Plane(euler=(35,0,0)),gs.materials.Rigid(needs_coup=True))
unrelated=scene.add_entity(gs.morphs.Box(pos=(0,0,-.9),size=(.1,.1,.1),fixed=True),gs.materials.Rigid(needs_coup=True))
cloth=scene.add_entity(gs.morphs.Mesh(file='meshes/cloth.obj',scale=.1,pos=(0,0,.3)),gs.materials.PBD.Cloth())
scene.build();cloth.fix_particles()
pbd=scene.pbd_solver;rigid=scene.rigid_solver;coupler=scene.sim.coupler
pbd._kernel_reorder_particles(0)
kw=dict(geoms_state=rigid.dyn_state.geoms,geoms_info=rigid.dyn_info.geoms,links_state=rigid.dyn_state.links,sdf_info=rigid.collider._sdf._sdf_info,rigid_info=rigid.rigid_info,collider_static_config=rigid.collider.collider_config)

@qd.data_oriented
class Probe:
 def __init__(self):self.solver=pbd
 @qd.kernel
 def seed(self,previous:qd.types.ndarray(),predicted:qd.types.ndarray(),free:qd.i32):
  idx=self.solver.particles_ng[0,0].reordered_idx
  self.solver.particles_reordered[idx,0].ipos=qd.Vector([previous[0],previous[1],previous[2]])
  self.solver.particles_reordered[idx,0].pos=qd.Vector([predicted[0],predicted[1],predicted[2]])
  self.solver.particles_reordered[idx,0].vel=(self.solver.particles_reordered[idx,0].pos-self.solver.particles_reordered[idx,0].ipos)/self.solver._substep_dt
  self.solver.particles_reordered[idx,0].free=free
 def read(self):
  idx=int(self.solver.particles_ng.reordered_idx.to_numpy()[0,0])
  return self.solver.particles_reordered.pos.to_numpy()[idx,0].copy(),self.solver.particles_reordered.vel.to_numpy()[idx,0].copy()

probe=Probe();angle=np.deg2rad(35);normal=np.array([0,-np.sin(angle),np.cos(angle)])
previous=.005*normal;predicted=previous+np.array([0,0,-9.81])*.01**2
probe.seed(previous,predicted,1);coupler.kernel_pbd_rigid_collide(**kw);basepos,basevel=probe.read()
np.testing.assert_allclose(np.dot(basepos,normal),.005,atol=1e-7)
for mu_s,mu_k in [(0,0),(1,.2),(.3,.2)]:
 extension=FixedBodyContactFriction(cloth,[plane],mu_static=mu_s,mu_kinetic=mu_k)
 probe.seed(previous,predicted,1);coupler.kernel_pbd_rigid_collide(**kw);pos,vel=probe.read()
 if mu_s==0:
  np.testing.assert_array_equal(pos,basepos);np.testing.assert_array_equal(vel,basevel)
 else:
  expected=basepos+coulomb_tangent_correction(basepos-previous,basepos-predicted,mu_s,mu_k)
  np.testing.assert_allclose(pos,expected,atol=2e-7)
  np.testing.assert_allclose(vel,(pos-previous)/.01,atol=2e-6)
  assert extension.metrics()['friction_applied_count_last_substep']==1
 print('NATIVE_INCLINE',mu_s,mu_k,'displacement',float(np.linalg.norm(pos-previous)),extension.metrics(),flush=True)
 extension.uninstall()
# A selected unrelated fixed body cannot add friction at the unselected plane.
extension=FixedBodyContactFriction(cloth,[unrelated],mu_static=1,mu_kinetic=.2)
probe.seed(previous,predicted,1);coupler.kernel_pbd_rigid_collide(**kw);pos,vel=probe.read()
np.testing.assert_array_equal(pos,basepos);np.testing.assert_array_equal(vel,basevel)
assert extension.metrics()['friction_applied_count_last_substep']==0
extension.uninstall()
# Gripper/fixed cloth particles must receive exactly native contact handling.
extension=FixedBodyContactFriction(cloth,[plane],mu_static=1,mu_kinetic=.2)
probe.seed(previous,predicted,0);coupler.kernel_pbd_rigid_collide(**kw);pos,vel=probe.read()
np.testing.assert_array_equal(pos,basepos);np.testing.assert_array_equal(vel,basevel)
assert extension.metrics()['friction_applied_count_last_substep']==0
extension.uninstall()
assert not hasattr(coupler,'_fixed_contact_friction_owner')
print('PASS native inclined-plane static/kinetic Coulomb, bitwise mu=0, selected-body isolation, pinned-particle exclusion, uninstall',flush=True)
