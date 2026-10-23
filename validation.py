"""Geometric checks for the image-only reconstruction, requiring only NumPy.

The sleeve check is an instantaneous geometric test, not proof of a collision-free
threading trajectory. A nonplanar or self-intersecting aperture is deliberately
reported as unreliable rather than optimistically declaring success.

Conventions: ``left`` denotes the hole with smaller REST world-x coordinate.
``arms[side]`` is ``(shoulder_xyz, hand_tip_xyz)`` in current world coordinates.
Faces and current/rest vertices must describe the SAME mesh after any remeshing.
"""

from collections import defaultdict

import numpy as np


def _faces_array(faces):
    raw = np.asarray(faces)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise ValueError("face indices must be integers")
    result = raw.astype(np.int64)
    if result.size and result.min() < 0:
        raise ValueError("face indices must be nonnegative")
    if np.any((result[:, 0] == result[:, 1]) | (result[:, 1] == result[:, 2])
              | (result[:, 2] == result[:, 0])):
        raise ValueError("a triangle contains repeated vertex indices")
    return result


def _unique_edges(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)


def _connected_component_count(faces):
    adjacency = defaultdict(list)
    edges, _ = _unique_edges(faces)
    for a, b in edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    remaining, component_count = set(adjacency), 0
    while remaining:
        stack = [remaining.pop()]
        component_count += 1
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return component_count


def extract_boundary_loops(faces):
    """Return deterministic ordered boundary rings, without repeating first IDs.

    Raises ValueError for nonmanifold edges, branching/open boundary vertices,
    repeated triangles, or invalid triangular indices. The loop orientation is
    arbitrary; callers must not infer a surface normal from its winding.
    """
    faces = _faces_array(faces)
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("duplicate triangles make the boundary ambiguous")
    edges, counts = _unique_edges(faces)
    if np.any(counts > 2):
        raise ValueError("nonmanifold edge belongs to more than two triangles")
    adjacency = defaultdict(list)
    for a, b in edges[counts == 1]:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("boundary is branching or is an open chain")
    remaining = set(adjacency)
    loops = []
    while remaining:
        start = min(remaining)
        previous, current = start, min(adjacency[start])
        loop = [start]
        visited = {start}
        while current != start:
            if current in visited:
                raise ValueError("boundary repeats a vertex before closing")
            loop.append(current)
            visited.add(current)
            neighbors = adjacency[current]
            next_vertex = neighbors[0] if neighbors[1] == previous else neighbors[1]
            previous, current = current, next_vertex
        if len(loop) < 3:
            raise ValueError("boundary has fewer than three vertices")
        remaining.difference_update(loop)
        loops.append(loop)
    return loops


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _point_segment_distance(point, a, b):
    direction = b - a
    denom = float(direction @ direction)
    if denom <= np.finfo(float).tiny:
        return float(np.linalg.norm(point - a))
    t = np.clip(float((point - a) @ direction) / denom, 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * direction)))


def _segments_intersect(a, b, c, d, tolerance):
    if (max(a[0], b[0]) < min(c[0], d[0]) - tolerance
            or max(c[0], d[0]) < min(a[0], b[0]) - tolerance
            or max(a[1], b[1]) < min(c[1], d[1]) - tolerance
            or max(c[1], d[1]) < min(a[1], b[1]) - tolerance):
        return False
    ab, cd = b - a, d - c
    values = (_cross2(ab, c - a), _cross2(ab, d - a),
              _cross2(cd, a - c), _cross2(cd, b - c))
    area_tol = tolerance * max(float(np.linalg.norm(ab)), float(np.linalg.norm(cd)), tolerance)
    if values[0] * values[1] < -(area_tol ** 2) and values[2] * values[3] < -(area_tol ** 2):
        return True
    return any((abs(value) <= area_tol and _point_segment_distance(p, q, r) <= tolerance)
               for value, p, q, r in ((values[0], c, a, b), (values[1], d, a, b),
                                      (values[2], a, c, d), (values[3], b, c, d)))


def _simple_polygon(points, tolerance):
    count = len(points)
    for i in range(count):
        if np.linalg.norm(points[(i + 1) % count] - points[i]) <= tolerance:
            return False
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            if _segments_intersect(points[i], points[(i + 1) % count],
                                   points[j], points[(j + 1) % count], tolerance):
                return False
    return True


def _point_in_polygon(point, polygon, tolerance):
    """Return (strictly_inside, on_boundary); ambiguous boundary is not inside."""
    inside = False
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if _point_segment_distance(point, a, b) <= tolerance:
            return False, True
        if (a[1] > point[1]) != (b[1] > point[1]):
            crossing_x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if point[0] < crossing_x:
                inside = not inside
    return inside, False


def _fit_aperture(points):
    center = points.mean(axis=0)
    centered = points - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[2].copy()
    # Canonical sign makes signed distances repeatable. It is not "inside/outside".
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    axis_u = vh[0]
    axis_v = np.cross(normal, axis_u)
    basis = np.column_stack((axis_u, axis_v))
    polygon = centered @ basis
    residual = centered @ normal
    diameter = float(np.linalg.norm(np.ptp(polygon, axis=0)))
    tolerance = max(1e-9, diameter * 1e-5)
    area = abs(float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                            - polygon[:, 1] * np.roll(polygon[:, 0], -1)))) / 2
    rms = float(np.sqrt(np.mean(residual ** 2)))
    maximum = float(np.max(np.abs(residual)))
    aspect = float(singular_values[1] / max(singular_values[0], 1e-15))
    reasons = []
    if diameter <= 1e-8 or area <= diameter * diameter * 1e-4 or aspect < 0.05:
        reasons.append("aperture_is_degenerate_or_nearly_collapsed")
    if rms > 0.05 * diameter or maximum > 0.10 * diameter:
        reasons.append("aperture_is_too_nonplanar_for_a_plane_crossing_test")
    if not _simple_polygon(polygon, tolerance):
        reasons.append("projected_boundary_is_self_intersecting_or_has_collapsed_edges")
    public = {
        "reliable": not reasons,
        "unreliable_reasons": reasons,
        "center": center.tolist(),
        "normal": normal.tolist(),
        "projected_area_m2": area,
        "projected_diameter_m": diameter,
        "planarity_rms_m": rms,
        "planarity_max_m": maximum,
        "planarity_rms_over_diameter": rms / diameter if diameter > 0 else None,
        "in_plane_singular_value_ratio": aspect,
    }
    return public, (center, normal, basis, polygon, tolerance)


def _arm_crossing(arm, fit, fit_reliable):
    shoulder, tip = np.asarray(arm, dtype=float)
    center, normal, basis, polygon, tolerance = fit
    ds, dt = float((shoulder - center) @ normal), float((tip - center) @ normal)
    endpoint_margin = max(tolerance, 1e-7)
    straddles = (ds > endpoint_margin and dt < -endpoint_margin) or (
        dt > endpoint_margin and ds < -endpoint_margin)
    denominator = ds - dt
    line_length = float(np.linalg.norm(tip - shoulder))
    parallel = abs(denominator) <= max(endpoint_margin, 1e-4 * line_length)
    point, fraction, inside, on_boundary = None, None, False, False
    if not parallel:
        fraction = ds / denominator
        point_array = shoulder + fraction * (tip - shoulder)
        point = point_array.tolist()
        inside, on_boundary = _point_in_polygon((point_array - center) @ basis, polygon, tolerance)
    passed = bool(fit_reliable and straddles and not parallel and inside and not on_boundary)
    return {
        "passed": passed,
        "shoulder_signed_distance_m": ds,
        "tip_signed_distance_m": dt,
        "endpoints_straddle_plane": bool(straddles),
        "line_nearly_parallel_to_plane": bool(parallel),
        "intersection_fraction_shoulder_to_tip": fraction,
        "intersection_position": point,
        "intersection_inside_projected_polygon": bool(inside),
        "intersection_on_boundary_ambiguous": bool(on_boundary),
    }


def analyze_cloth(rest_positions, faces, current_positions, arms, torso_center, torso_radii):
    """Return JSON-serializable sleeve, vertex penetration, and edge strain metrics.

    ``threading_passed`` fails closed unless the mesh has exactly three manifold
    boundary loops, both apertures are reliable, and each arm crosses exactly its
    matching aperture. Planarity tolerances are 5% RMS / 10% maximum of aperture
    diameter. These are explicit geometric heuristics, not material parameters.

    Ellipsoid depth is distance to its surface along a radial ray; it is generally
    an upper bound on the true shortest penetration depth. Only vertices are
    tested, so this does not certify triangle-interior contact or all body parts.
    """
    rest = np.asarray(rest_positions, dtype=float)
    current = np.asarray(current_positions, dtype=float)
    faces = _faces_array(faces)
    if rest.ndim != 2 or rest.shape[1] != 3 or current.shape != rest.shape or len(rest) < 3:
        raise ValueError("rest and current positions must have matching shape (N, 3)")
    if not np.all(np.isfinite(rest)) or not np.all(np.isfinite(current)):
        raise ValueError("mesh positions must be finite")
    if not len(faces) or int(faces.max()) >= len(rest):
        raise ValueError("faces must be nonempty and reference existing vertices")
    center, radii = np.asarray(torso_center, dtype=float), np.asarray(torso_radii, dtype=float)
    if center.shape != (3,) or radii.shape != (3,) or not np.all(np.isfinite(center)) \
            or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("torso center/radii must be finite length-3 vectors with positive radii")
    for side in ("left", "right"):
        if side not in arms or np.asarray(arms[side]).shape != (2, 3) \
                or not np.all(np.isfinite(np.asarray(arms[side], dtype=float))):
            raise ValueError("arms must contain finite left/right (shoulder, tip) pairs")

    result = {
        "threading_passed": False,
        "reliable": False,
        "scope": "Instantaneous geometry only; collision-free threading history is not certified.",
        "side_convention": "left/right are ordered by increasing rest world-x, not anatomical side",
        "topology": {"passed": False, "boundary_loop_count": None, "error": None},
        "sleeves": {},
    }
    try:
        loops = extract_boundary_loops(faces)
        result["topology"]["boundary_loop_count"] = len(loops)
        component_count = _connected_component_count(faces)
        result["topology"]["connected_component_count"] = component_count
        if component_count != 1:
            result["topology"]["error"] = "expected one connected cloth surface"
        elif len(loops) != 3:
            result["topology"]["error"] = "expected exactly one outer boundary and two sleeve holes"
        else:
            rest_fits = [_fit_aperture(rest[loop])[0] for loop in loops]
            outer_index = int(np.argmax([fit["projected_area_m2"] for fit in rest_fits]))
            holes = [loop for i, loop in enumerate(loops) if i != outer_index]
            holes.sort(key=lambda ids: float(rest[ids, 0].mean()))
            separation = float(rest[holes[1], 0].mean() - rest[holes[0], 0].mean())
            _, outer_fit = _fit_aperture(rest[loops[outer_index]])
            outer_center, outer_normal, outer_basis, outer_polygon, outer_tol = outer_fit
            holes_inside_outer = all(
                _point_in_polygon((point - outer_center) @ outer_basis, outer_polygon, outer_tol)[0]
                and abs(float((point - outer_center) @ outer_normal)) <= max(outer_tol, 1e-7)
                for loop in holes for point in rest[loop])
            if separation <= max(float(np.ptp(rest[:, 0])) * 1e-5, 1e-9):
                result["topology"]["error"] = "sleeve ordering by rest x is ambiguous"
            elif not holes_inside_outer:
                result["topology"]["error"] = "rest sleeve rings must lie strictly inside the planar outer boundary"
            elif not all(fit["reliable"] for fit in rest_fits):
                result["topology"]["error"] = "rest boundaries are not reliable simple planar rings"
            else:
                result["topology"].update({"passed": True, "outer_boundary_vertex_count": len(loops[outer_index])})
                for side, loop in zip(("left", "right"), holes):
                    public, fit = _fit_aperture(current[loop])
                    crossings = {arm_side: _arm_crossing(arms[arm_side], fit, public["reliable"])
                                 for arm_side in ("left", "right")}
                    other_side = "right" if side == "left" else "left"
                    result["sleeves"][side] = {
                        **public,
                        "boundary_vertex_count": len(loop),
                        "boundary_vertex_indices": [int(i) for i in loop],
                        "rest_center": rest[loop].mean(axis=0).tolist(),
                        "arm_crossings": crossings,
                        "passed": bool(crossings[side]["passed"] and not crossings[other_side]["passed"]),
                    }
                result["reliable"] = all(sleeve["reliable"] for sleeve in result["sleeves"].values())
                result["threading_passed"] = bool(result["reliable"] and all(
                    sleeve["passed"] for sleeve in result["sleeves"].values()))
    except ValueError as error:
        result["topology"]["error"] = str(error)

    displacement = current - center
    radial_norm = np.linalg.norm(displacement / radii, axis=1)
    inside = radial_norm < 1.0
    depths = np.zeros(len(current), dtype=float)
    noncentral = inside & (radial_norm > 1e-12)
    depths[noncentral] = np.linalg.norm(displacement[noncentral], axis=1) * (
        1.0 / radial_norm[noncentral] - 1.0)
    depths[inside & ~noncentral] = float(np.min(radii))
    result["torso_penetration"] = {
        "method": "vertex-to-ellipsoid surface distance along radial rays; not exact closest distance",
        "scope": "vertices only, torso ellipsoid only; triangle interiors and other body parts not certified",
        "vertex_count": int(len(current)),
        "inside_vertex_count": int(np.count_nonzero(inside)),
        "max_radial_depth_m": float(np.max(depths)),
        "mean_radial_depth_m": float(np.mean(depths)),
        "p95_radial_depth_m": float(np.quantile(depths, 0.95)),
    }
    edges, _ = _unique_edges(faces)
    rest_lengths = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    lengths = np.linalg.norm(current[edges[:, 1]] - current[edges[:, 0]], axis=1)
    valid = rest_lengths > max(float(rest_lengths.max()) * 1e-12, 1e-12)
    ratios = lengths[valid] / rest_lengths[valid]
    labels, quantiles = ("min", "p05", "median", "p95", "p99", "max"), (0, .05, .5, .95, .99, 1)
    result["edge_stretch"] = {
        "edge_count": int(len(edges)),
        "zero_rest_length_edge_count": int(np.count_nonzero(~valid)),
        "reliable": bool(np.all(valid)),
        "length_ratio": {label: float(value) for label, value in zip(labels, np.quantile(ratios, quantiles))}
        if len(ratios) else None,
        "max_extension_strain": max(0.0, float(np.max(ratios)) - 1.0) if len(ratios) else None,
    }
    if not np.all(valid):
        result["reliable"] = False
        result["threading_passed"] = False
    return result
