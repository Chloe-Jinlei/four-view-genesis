"""Contact-triggered, breakable seam approximation for native Genesis PBD cloth.

This is a macroscopic Velcro approximation, not a hook-and-loop microstructure
model.  Each matched boundary pair receives an XPBD distance constraint only
after proximity and relative-speed checks pass.  No target garment shape is
imposed, and no cloth vertex outside the supplied pairs is written here.

Genesis 1.4.3 has no public runtime seam API.  A removable, per-solver hook adds
this topology constraint immediately before native spatial hashing.  Native
self-collision, velocity reconstruction, and rigid coupling then run normally.
This private integration point must be reviewed when upgrading Genesis.  It is
not differentiable and its state is not included in Genesis checkpoints.
"""

import numpy as np
import quadrants as qd

import genesis as gs


def _particle_ids(values, count, name):
    """Validate local particle IDs without depending on an initialized engine."""
    ids = np.asarray(values)
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError(f"{name} must be a one-dimensional integer array.")
    if ids.size and (ids.min() < 0 or ids.max() >= count):
        raise ValueError(f"{name} contains an out-of-range particle ID.")
    return np.unique(ids.astype(np.int32))


def boundary_strips(initial_positions, *, axis=0, width=0.028):
    """Return material IDs on both full-width end strips of a flat garment.

    Membership is selected ONCE from the remeshed initial state, not from the
    moving cloth.  Width is a reconstruction assumption (28 mm by default).
    """
    positions = np.asarray(initial_positions)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("initial_positions must have nonempty shape (N, 3).")
    if not np.isfinite(positions).all() or axis not in (0, 1, 2):
        raise ValueError("Require finite positions and a valid XYZ axis.")
    values = positions[:, axis]
    if not np.isfinite(width) or width <= 0 or 2 * width >= np.ptp(values):
        raise ValueError("width must select distinct opposite strips.")
    return (
        np.flatnonzero(values <= values.min() + width).astype(np.int32),
        np.flatnonzero(values >= values.max() - width).astype(np.int32),
    )


def select_nearest_contacts(
    positions, velocities, left_strip, right_strip, *, max_pairs,
    reserved=(), max_distance=0.020, max_relative_speed=0.10,
):
    """Pure NumPy greedy matching of disjoint, nearby, slow strip contacts.

    Candidates are ordered by distance, then particle IDs for deterministic
    ties.  An endpoint is used at most once.  This is nearest-first contact
    assignment, not a global maximum-cardinality matching.  Material-Y order
    need not agree after folding.  No state or particle position is modified.

    The 20 mm upper bound is deliberate: this matcher must not create bonds
    across a non-contacting gap.  Normals, hook/loop direction, contact pressure,
    and area coverage are not inferred from point proximity alone.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or velocities.shape != positions.shape:
        raise ValueError("positions and velocities must share shape (N, 3).")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("Contact selection requires finite positions and velocities.")
    if not isinstance(max_pairs, (int, np.integer)) or max_pairs < 0:
        raise ValueError("max_pairs must be a nonnegative integer.")
    if not np.isfinite(max_distance) or not 0 < max_distance <= 0.020:
        raise ValueError("Nearest contact distance must be in (0, 0.020] metres.")
    if not np.isfinite(max_relative_speed) or max_relative_speed < 0:
        raise ValueError("max_relative_speed must be finite and nonnegative.")
    left = _particle_ids(left_strip, len(positions), "left_strip")
    right = _particle_ids(right_strip, len(positions), "right_strip")
    blocked = _particle_ids(reserved, len(positions), "reserved")
    if np.intersect1d(left, right).size:
        raise ValueError("The opposing material strips must not share endpoints.")
    left = left[~np.isin(left, blocked)]
    right = right[~np.isin(right, blocked)]
    if max_pairs == 0 or not len(left) or not len(right):
        return np.empty((0, 2), dtype=np.int32)
    delta = positions[left, None, :] - positions[None, right, :]
    relative_velocity = velocities[left, None, :] - velocities[None, right, :]
    distances_squared = np.einsum('ijk,ijk->ij', delta, delta)
    speeds_squared = np.einsum('ijk,ijk->ij', relative_velocity, relative_velocity)
    li, ri = np.nonzero(
        (distances_squared <= max_distance ** 2)
        & (speeds_squared <= max_relative_speed ** 2)
    )
    order = np.lexsort((right[ri], left[li], distances_squared[li, ri]))
    used, chosen = set(), []
    for candidate in order:
        a, b = int(left[li[candidate]]), int(right[ri[candidate]])
        if a in used or b in used:
            continue
        chosen.append((a, b))
        used.update((a, b))
        if len(chosen) == max_pairs:
            break
    return np.asarray(chosen, dtype=np.int32).reshape(-1, 2)


@qd.data_oriented
class VelcroSeam:
    """Install after ``scene.build()``, using local particle IDs on ``cloth``.

    Example::

        seam = VelcroSeam(cloth, pairs, enabled=False)
        seam.enable()  # begin testing real contact near the closure phase
        scene.step()  # no separate before_step/after_step call is needed
        record = seam.metrics()

    ``pairs`` must have shape (N, 2) with every endpoint appearing once.  Match
    two physical boundary strips by their vertical material coordinates after
    Genesis has remeshed the cloth; original OBJ vertex IDs are not valid IDs.
    ``break_extension`` is separation beyond ``rest_gap``, in metres.  A broken
    pair stays open unless ``allow_reattach=True``.  ``enable(False)`` only blocks
    new latches; ``release_all()`` opens existing latches.

    ``rest_gap`` defaults to the solver particle diameter so seam constraints
    do not directly oppose Genesis's particle self-collision separation.
    Surface-normal alignment, pressure, tangential friction and peel mechanics
    are not represented by this scalar-distance approximation.

    Optional ``contact_mode='nearest'`` requires explicit ``left_strip`` and
    ``right_strip`` material IDs, for example from ``boundary_strips(rest)``.
    Call ``before_step()`` once before EACH scene step.  The supplied pairs then
    define only the maximum number of bond slots.  Never-latched inactive slots
    are reassigned to nearest current contacts; previously latched identities
    are retained permanently, including broken bonds.  This CPU matching mode
    currently supports one environment and distances no greater than 20 mm.
    """

    def __init__(
        self,
        cloth,
        pairs,
        *,
        attach_distance=0.012,
        max_attach_speed=0.10,
        break_extension=0.035,
        rest_gap=None,
        compliance=1e-7,
        iterations=2,
        relaxation=1.0,
        enabled=True,
        allow_reattach=False,
        contact_mode="fixed",
        left_strip=None,
        right_strip=None,
    ):
        if not cloth.is_built:
            raise ValueError("Construct VelcroSeam after scene.build().")
        if not isinstance(cloth.material, gs.materials.PBD.Cloth):
            raise TypeError("VelcroSeam requires native Genesis PBD.Cloth.")
        if cloth.solver._sim.requires_grad:
            raise ValueError("The Velcro extension does not support differentiable simulation.")
        raw_pairs = np.asarray(pairs)
        if raw_pairs.ndim != 2 or raw_pairs.shape[1] != 2 or not len(raw_pairs):
            raise ValueError("pairs must be a nonempty (N, 2) array.")
        if not np.issubdtype(raw_pairs.dtype, np.integer):
            raise TypeError("pairs must contain integer local particle IDs.")
        local_pairs = raw_pairs.astype(np.int32)
        if local_pairs.min() < 0 or local_pairs.max() >= cloth.n_particles:
            raise ValueError("A seam endpoint is outside this cloth's particle range.")
        if np.unique(local_pairs).size != local_pairs.size:
            raise ValueError("Seam pairs must be disjoint: each endpoint appears only once.")
        self.cloth = cloth
        self.solver = cloth.solver
        if getattr(self.solver, "_velcro_seam_owner", None) is not None:
            raise ValueError("Only one VelcroSeam per PBD solver is supported.")
        self.local_pairs = local_pairs.copy()
        self.n_pairs = len(local_pairs)
        self.n_envs = self.solver._B
        self.attach_distance = float(attach_distance)
        self.max_attach_speed = float(max_attach_speed)
        self.break_extension = float(break_extension)
        self.rest_gap = float(self.solver.particle_size if rest_gap is None else rest_gap)
        self.compliance = float(compliance)
        self.iterations = int(iterations)
        self.relaxation = float(relaxation)
        self.allow_reattach = bool(allow_reattach)
        if not all(np.isfinite(v) for v in (
            self.attach_distance, self.max_attach_speed, self.break_extension,
            self.rest_gap, self.compliance, self.relaxation,
        )):
            raise ValueError("Seam parameters must be finite.")
        if not 0 < self.rest_gap <= self.attach_distance:
            raise ValueError("Require 0 < rest_gap <= attach_distance.")
        if self.rest_gap < self.solver.particle_size * (1 - 1e-6):
            raise ValueError("rest_gap must be at least PBD particle_size to respect self collision.")
        if self.max_attach_speed < 0 or self.break_extension <= 0 or self.compliance < 0:
            raise ValueError("Speed/compliance must be nonnegative; break_extension positive.")
        if self.iterations < 1 or not 0 < self.relaxation <= 1:
            raise ValueError("Require iterations >= 1 and 0 < relaxation <= 1.")
        if contact_mode not in ("fixed", "nearest"):
            raise ValueError("contact_mode must be 'fixed' or 'nearest'.")
        self.contact_mode = contact_mode
        self._last_candidate_count = 0
        if contact_mode == "nearest":
            if self.n_envs != 1:
                raise ValueError("Nearest-contact matching currently supports one environment.")
            if self.attach_distance > 0.020:
                raise ValueError("Nearest-contact attachment distance cannot exceed 20 mm.")
            if left_strip is None or right_strip is None:
                raise ValueError("Nearest mode requires explicit left_strip and right_strip IDs.")
            self.left_strip = _particle_ids(left_strip, cloth.n_particles, "left_strip")
            self.right_strip = _particle_ids(right_strip, cloth.n_particles, "right_strip")
            if not len(self.left_strip) or not len(self.right_strip):
                raise ValueError("Both opposing material strips must be nonempty.")
            if np.intersect1d(self.left_strip, self.right_strip).size:
                raise ValueError("The opposing material strips must be disjoint.")

        self._pairs = qd.field(dtype=gs.qd_int, shape=(self.n_pairs, 2))
        self._pairs.from_numpy(local_pairs + cloth._particle_start)
        state_shape = (self.n_pairs, self.n_envs)
        self._active = qd.field(dtype=gs.qd_int, shape=state_shape)
        self._attach_count = qd.field(dtype=gs.qd_int, shape=state_shape)
        self._break_count = qd.field(dtype=gs.qd_int, shape=state_shape)
        self._lambda = qd.field(dtype=gs.qd_float, shape=state_shape)
        self._slot_enabled = qd.field(dtype=gs.qd_int, shape=(self.n_pairs,))
        self._enabled = qd.field(dtype=gs.qd_int, shape=())
        self._active.fill(0)
        self._attach_count.fill(0)
        self._break_count.fill(0)
        self._lambda.fill(0)
        self._slot_enabled.fill(int(contact_mode == "fixed"))
        self.enable(enabled)
        self._original_reorder = self.solver._kernel_reorder_particles

        # Python dispatch calls this immediately after native stretch/bending.
        # Native hashing and contact see the corrected positions, and native
        # velocity reconstruction uses the unchanged beginning-of-substep ipos.
        def reorder_with_seam(f):
            self._update_and_solve()
            self._original_reorder(f)

        self._hook = reorder_with_seam
        self.solver._kernel_reorder_particles = self._hook
        self.solver._velcro_seam_owner = self

    def enable(self, enabled=True):
        """Enable or disable detection of new contacts, retaining existing bonds."""
        self._enabled[None] = int(bool(enabled))

    def before_step(self):
        """Refresh nearest contact candidates; return candidate-slot count.

        This changes only bookkeeping, never cloth positions or velocities.
        Existing bonds are reserved before greedy matching, and native substeps
        independently recheck proximity/speed before activating a fresh bond.
        Unfilled slots are disabled, so stale placeholder IDs cannot attach or
        compete with valid bonds.  Fixed matching mode is a no-op.
        """
        if self.contact_mode == "fixed":
            return 0
        attached = self._attach_count.to_numpy()[:, 0] > 0
        active = self._active.to_numpy()[:, 0] != 0
        retained = attached | active
        free_slots = np.flatnonzero(~retained)
        slot_enabled = retained.astype(np.int32)
        chosen = np.empty((0, 2), dtype=np.int32)
        if self._enabled[None] and len(free_slots):
            positions = self.cloth.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
            velocities = self.cloth.get_particles_vel().detach().cpu().numpy().reshape(-1, 3)
            chosen = select_nearest_contacts(
                positions, velocities, self.left_strip, self.right_strip,
                max_pairs=len(free_slots), reserved=self.local_pairs[retained].ravel(),
                max_distance=self.attach_distance, max_relative_speed=self.max_attach_speed,
            )
            assigned = free_slots[:len(chosen)]
            self.local_pairs[assigned] = chosen
            slot_enabled[assigned] = 1
        self._pairs.from_numpy(self.local_pairs + self.cloth._particle_start)
        self._slot_enabled.from_numpy(slot_enabled)
        self._last_candidate_count = len(chosen)
        return len(chosen)

    def release_all(self):
        """Explicitly open all current bonds without counting this as breakage."""
        self._active.fill(0)
        self._lambda.fill(0)

    def reset(self):
        """Clear seam state after a scene reset; preserves the enabled setting."""
        self.release_all()
        self._attach_count.fill(0)
        self._break_count.fill(0)
        if self.contact_mode == "nearest":
            self._slot_enabled.fill(0)
            self._last_candidate_count = 0

    def uninstall(self):
        """Restore native behavior; no engine installation files are modified."""
        if getattr(self.solver, "_velcro_seam_owner", None) is self:
            if self.solver._kernel_reorder_particles is not self._hook:
                raise RuntimeError("Another extension replaced the seam hook; restore it first.")
            self.solver._kernel_reorder_particles = self._original_reorder
            del self.solver._velcro_seam_owner

    @qd.kernel
    def _update_and_solve(self):
        for pair, env in qd.ndrange(self.n_pairs, self.n_envs):
            a = self._pairs[pair, 0]
            b = self._pairs[pair, 1]
            live = self.solver.particles_ng[a, env].active and self.solver.particles_ng[b, env].active
            gap = (self.solver.particles[a, env].pos - self.solver.particles[b, env].pos).norm()
            speed = (self.solver.particles[a, env].vel - self.solver.particles[b, env].vel).norm()
            self._lambda[pair, env] = 0.0

            if self._active[pair, env] != 0:
                if not live or gap - self.rest_gap > self.break_extension:
                    self._active[pair, env] = 0
                    self._break_count[pair, env] += 1
            elif (
                live and self._enabled[None] != 0
                and self._slot_enabled[pair] != 0
                and gap <= self.attach_distance
                and speed <= self.max_attach_speed
                and (self.allow_reattach or self._break_count[pair, env] == 0)
            ):
                self._active[pair, env] = 1
                self._attach_count[pair, env] += 1

            if self._active[pair, env] != 0:
                # Fixed/grasped particles have zero inverse mass, so this seam
                # never overrides a gripper attachment or an explicit pin.
                wa = qd.cast(self.solver.particles[a, env].free, gs.qd_float) / self.solver.particles_info[a].mass
                wb = qd.cast(self.solver.particles[b, env].free, gs.qd_float) / self.solver.particles_info[b].mass
                alpha = self.compliance / (self.solver._substep_dt ** 2)
                if wa + wb > 0.0:
                    for _ in qd.static(range(self.iterations)):
                        delta = self.solver.particles[a, env].pos - self.solver.particles[b, env].pos
                        distance = delta.norm()
                        if distance > gs.EPS:
                            dlambda = (
                                -(distance - self.rest_gap) - alpha * self._lambda[pair, env]
                            ) / (wa + wb + alpha) * self.relaxation
                            self._lambda[pair, env] += dlambda
                            correction = dlambda * delta / distance
                            self.solver.particles[a, env].pos += wa * correction
                            self.solver.particles[b, env].pos -= wb * correction

    def metrics(self, env=0):
        """Return JSON-compatible current gaps and cumulative event counts."""
        env = int(env)
        if not 0 <= env < self.n_envs:
            raise IndexError("Environment index is out of bounds.")
        positions = self.cloth.get_particles_pos().detach().cpu().numpy()
        if positions.ndim == 3:
            positions = positions[env]
        gaps = np.linalg.norm(
            positions[self.local_pairs[:, 0]] - positions[self.local_pairs[:, 1]], axis=1,
        )
        active = self._active.to_numpy()[:, env].astype(bool)
        active_gaps = gaps[active]
        return {
            "pair_count": self.n_pairs,
            "active_count": int(active.sum()),
            "attached_total": int(self._attach_count.to_numpy()[:, env].sum()),
            "breaks_total": int(self._break_count.to_numpy()[:, env].sum()),
            "mean_gap_m": float(gaps.mean()),
            "max_gap_m": float(gaps.max()),
            "mean_active_gap_m": float(active_gaps.mean()) if len(active_gaps) else None,
            "max_active_gap_m": float(active_gaps.max()) if len(active_gaps) else None,
            "gaps_m": gaps.tolist(),
            "active_pairs": np.flatnonzero(active).tolist(),
            "attachment_enabled": bool(self._enabled[None]),
            "rest_gap_m": self.rest_gap,
            "contact_mode": self.contact_mode,
            "enabled_slot_count": int(self._slot_enabled.to_numpy().sum()),
            "candidate_count_last_update": self._last_candidate_count,
            "pair_particles_local": self.local_pairs.tolist(),
        }


def match_boundary_pairs(initial_positions, axis=0, width=0.015, order_axis=1, max_pairs=None):
    """Match the opposite flat-cloth end strips after native remeshing.

    Selects points close to the minimum/maximum of ``axis`` and samples both
    strips in ascending ``order_axis`` order.  This helper assumes a rectangular
    flat garment; pass explicit IDs for a different pattern or strip topology.
    """
    positions = np.asarray(initial_positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("initial_positions must have shape (N, 3).")
    if axis == order_axis or axis not in (0, 1, 2) or order_axis not in (0, 1, 2):
        raise ValueError("axis and order_axis must be different XYZ axes.")
    values = positions[:, axis]
    if width <= 0 or 2 * width >= np.ptp(values):
        raise ValueError("width must select distinct narrow opposite strips.")
    left = np.flatnonzero(values <= values.min() + width)
    right = np.flatnonzero(values >= values.max() - width)
    left = left[np.argsort(positions[left, order_axis], kind="stable")]
    right = right[np.argsort(positions[right, order_axis], kind="stable")]
    n = min(len(left), len(right))
    if max_pairs is not None:
        if int(max_pairs) < 1:
            raise ValueError("max_pairs must be positive.")
        n = min(n, int(max_pairs))
    if not n:
        raise ValueError("No boundary particles were found.")
    return np.column_stack((
        left[np.rint(np.linspace(0, len(left) - 1, n)).astype(int)],
        right[np.rint(np.linspace(0, len(right) - 1, n)).astype(int)],
    )).astype(np.int32)
