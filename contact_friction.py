"""Optional local Coulomb friction for Genesis 1.4.3 native PBD contact.

simulate.py installs this extension when bear_contact_friction is configured.
It restricts friction to free
particles contacting exactly one supplied, fixed rigid geometry in a substep.
Native normal projection supplies the normal-impulse proxy. The tangential
position correction is bounded by mu * normal displacement; velocity is then
reconstructed from the same substep ipos. There is no adhesion or global drag.

Multi-geometry contacts are deliberately skipped: the norm of a sum of normal
projections is not a reliable Coulomb budget for several different normals.
This conservative approximation therefore under-models contact at overlapping
bear parts. It also assumes the supplied fixed bodies remain stationary.
Private per-instance hooks are removable; no installed engine files change.
"""
import numpy as np
import quadrants as qd
import genesis as gs
from genesis.utils import array_class, sdf


def _coefficients(mu_static, mu_kinetic):
    values = np.asarray([mu_static, mu_kinetic], dtype=float)
    if not np.isfinite(values).all() or not 0 <= values[1] <= values[0]:
        raise ValueError("Require finite mu_static >= mu_kinetic >= 0.")
    return float(values[0]), float(values[1])


def coulomb_tangent_correction(displacement, normal_projection, mu_static, mu_kinetic):
    """Pure NumPy reference for one locally planar, stationary-body contact.

    ``displacement`` is the post-normal-projection position minus substep ipos.
    ``normal_projection`` is the native outward position correction. Returning
    zero for zero normal correction prevents any friction without compression.
    """
    mu_static, mu_kinetic = _coefficients(mu_static, mu_kinetic)
    displacement = np.asarray(displacement, dtype=float)
    normal_projection = np.asarray(normal_projection, dtype=float)
    if displacement.shape != (3,) or normal_projection.shape != (3,):
        raise ValueError("Expected two XYZ vectors.")
    if not np.isfinite(displacement).all() or not np.isfinite(normal_projection).all():
        raise ValueError("Contact vectors must be finite.")
    dn = np.linalg.norm(normal_projection)
    if mu_static == 0 or dn == 0:
        return np.zeros(3)
    normal = normal_projection / dn
    tangent = displacement - normal * np.dot(displacement, normal)
    distance = np.linalg.norm(tangent)
    if distance <= mu_static * dn:
        return -tangent
    return -tangent * min(mu_kinetic * dn / distance, 1.0)


@qd.data_oriented
class FixedBodyContactFriction:
    """Construct after build: ``FixedBodyContactFriction(cloth, [bear, ...])``.

    Only supplied fixed, coupled geometries are eligible. Table and robot
    contact receive no friction unless explicitly supplied (moving geometries
    are rejected). Static/kinetic defaults are assumed, not image observations.
    No before/after-step call is needed. ``uninstall()`` restores native dispatch.
    Zero coefficients or ``enabled=False`` call the original kernel directly,
    guaranteeing exact frictionless identity without extra particle writes.

    The hook queries the same native SDF in original geometry order to identify
    isolated contacts, calls native collision, then corrects tangential motion.
    It skips a correction that would enter another coupled geometry and uses
    native SDF normal projection if surface curvature requires it. Extra SDF
    queries add cost. Not differentiable; no automatic checkpoint persistence.
    """
    def __init__(self, cloth, fixed_entities, *, mu_static=.35, mu_kinetic=.25, enabled=True):
        if not cloth.is_built or not isinstance(cloth.material, gs.materials.PBD.Cloth):
            raise ValueError("Require a built native Genesis PBD cloth.")
        self.pbd = cloth.solver
        self.coupler = self.pbd._sim.coupler
        self.rigid = self.coupler.rigid_solver
        if self.pbd._sim.requires_grad or not self.coupler._rigid_pbd:
            raise ValueError("Require nondifferentiable native PBD/rigid coupling.")
        if getattr(self.coupler, '_fixed_contact_friction_owner', None) is not None:
            raise ValueError("Only one friction extension per coupler is supported.")
        self.mu_static, self.mu_kinetic = _coefficients(mu_static, mu_kinetic)
        selected = np.zeros(self.rigid.n_geoms, dtype=np.int32)
        for entity in fixed_entities:
            if not entity.is_built or entity.solver is not self.rigid:
                raise ValueError("Contact bodies must belong to this scene's rigid solver.")
            for geom in entity.geoms:
                if not geom.is_fixed:
                    raise ValueError("Contact friction supports stationary fixed geometry only.")
                if geom.needs_coup:
                    selected[geom.idx] = 1
        if not selected.any():
            raise ValueError("Supply at least one fixed geometry with needs_coup=True.")
        self.enabled = bool(enabled)
        self.n_particles = cloth.n_particles
        self.particle_start = cloth._particle_start
        self.n_envs = self.pbd._B
        self._selected = qd.field(dtype=gs.qd_int, shape=(self.rigid.n_geoms,))
        self._selected.from_numpy(selected)
        shape = (self.n_particles, self.n_envs)
        self._before = qd.Vector.field(3, dtype=gs.qd_float, shape=shape)
        self._contact_geom = qd.field(dtype=gs.qd_int, shape=shape)
        self._contact_geom.fill(-1)
        self._applied = qd.field(dtype=gs.qd_int, shape=shape)
        self._correction = qd.field(dtype=gs.qd_float, shape=shape)
        self._original = self.coupler.kernel_pbd_rigid_collide

        def collide_with_friction(*args, **kwargs):
            if not self.enabled or self.mu_static == 0:
                return self._original(*args, **kwargs)
            self._snapshot_contacts(*args, **kwargs)
            self._original(*args, **kwargs)
            self._apply_friction(*args, **kwargs)

        self._hook = collide_with_friction
        self.coupler.kernel_pbd_rigid_collide = self._hook
        self.coupler._fixed_contact_friction_owner = self

    def uninstall(self):
        if getattr(self.coupler, '_fixed_contact_friction_owner', None) is self:
            if self.coupler.kernel_pbd_rigid_collide is not self._hook:
                raise RuntimeError("Another extension replaced the hook; restore it first.")
            self.coupler.kernel_pbd_rigid_collide = self._original
            del self.coupler._fixed_contact_friction_owner

    def metrics(self):
        return {
            'enabled': self.enabled,
            'mu_static': self.mu_static, 'mu_kinetic': self.mu_kinetic,
            'isolated_contact_count_last_substep': int((self._contact_geom.to_numpy() >= 0).sum()),
            'friction_applied_count_last_substep': int(self._applied.to_numpy().sum()),
            'max_correction_m_last_substep': float(self._correction.to_numpy().max()),
        }

    @qd.kernel
    def _snapshot_contacts(
        self, geoms_state: array_class.GeomsState, geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState, sdf_info: array_class.SDFInfo,
        rigid_info: array_class.RigidInfo, collider_static_config: qd.template(),
    ):
        for local, env in qd.ndrange(self.n_particles, self.n_envs):
            original = local + self.particle_start
            self._contact_geom[local, env] = -1
            self._applied[local, env] = 0
            self._correction[local, env] = 0
            if self.pbd.particles_ng[original, env].active:
                idx = self.pbd.particles_ng[original, env].reordered_idx
                position = self.pbd.particles_reordered[idx, env].pos
                self._before[local, env] = position
                if self.pbd.particles_reordered[idx, env].free:
                    count, last_geom = 0, -1
                    # Reproduce native normal-projection order only to identify
                    # contacts; no particle writes are made by this pass.
                    for geom in range(self.rigid.n_geoms):
                        if geoms_info.needs_coup[geom]:
                            distance = sdf.sdf_func_world(geom, env, position, geoms_state, geoms_info, sdf_info)
                            if distance < self.pbd.particle_size / 2:
                                normal = sdf.sdf_func_normal_world(geom, env, position, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config)
                                position += normal * (self.pbd.particle_size / 2 - distance)
                                count += 1
                                last_geom = geom
                    if count == 1:
                        if self._selected[last_geom] != 0:
                            self._contact_geom[local, env] = last_geom

    @qd.kernel
    def _apply_friction(
        self, geoms_state: array_class.GeomsState, geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState, sdf_info: array_class.SDFInfo,
        rigid_info: array_class.RigidInfo, collider_static_config: qd.template(),
    ):
        for local, env in qd.ndrange(self.n_particles, self.n_envs):
            geom = self._contact_geom[local, env]
            if geom >= 0:
                original = local + self.particle_start
                idx = self.pbd.particles_ng[original, env].reordered_idx
                position = self.pbd.particles_reordered[idx, env].pos
                ipos = self.pbd.particles_reordered[idx, env].ipos
                projection = position - self._before[local, env]
                dn = projection.norm()
                if dn > gs.EPS:
                    normal = projection / dn
                    displacement = position - ipos
                    tangent = displacement - normal * displacement.dot(normal)
                    distance_t = tangent.norm()
                    if distance_t > gs.EPS:
                        amount = qd.min(self.mu_kinetic * dn, distance_t)
                        if distance_t <= self.mu_static * dn:
                            amount = distance_t
                        corrected = position - tangent * (amount / distance_t)
                        # Keep the normal constraint valid on curved surfaces.
                        distance = sdf.sdf_func_world(geom, env, corrected, geoms_state, geoms_info, sdf_info)
                        if distance < self.pbd.particle_size / 2:
                            outward = sdf.sdf_func_normal_world(geom, env, corrected, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config)
                            corrected += outward * (self.pbd.particle_size / 2 - distance)
                        valid = True
                        for other in range(self.rigid.n_geoms):
                            if other != geom and geoms_info.needs_coup[other]:
                                dother = sdf.sdf_func_world(other, env, corrected, geoms_state, geoms_info, sdf_info)
                                if dother < self.pbd.particle_size / 2:
                                    valid = False
                        if valid and amount > 0:
                            self.pbd.particles_reordered[idx, env].pos = corrected
                            self.pbd.particles_reordered[idx, env].vel = (corrected - ipos) / self.pbd._substep_dt
                            self._applied[local, env] = 1
                            self._correction[local, env] = amount
