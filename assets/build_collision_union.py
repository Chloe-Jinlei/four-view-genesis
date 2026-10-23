#!/usr/bin/env python3
"""Build the estimated teddy's union collision surface at unchanged dimensions.

Uses only this reconstruction's manifest. The implicit union removes surfaces
inside overlapping ellipsoids. It does not inflate the bear, close real gaps,
or alter visual assets. Marching-cubes discretization is an approximation of
the manifest's analytic ellipsoids, rather than an exact boolean of the
faceted visual OBJ files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
import trimesh


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=.0025, help="Uniform field spacing in metres")
    args = parser.parse_args()
    if not .0025 <= args.spacing <= .003:
        raise ValueError("Use the requested 2.5–3.0 mm grid spacing")
    manifest_path = ROOT / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    selected = [p for p in manifest["bear"]["parts"] if p["collision"] or p["name"] == "belly_patch"]
    shapes = []
    for p in selected:
        if p["geometry"] != "ellipsoid":
            raise ValueError(f"Unsupported collision geometry: {p['name']}")
        center = np.asarray(p["position"], dtype=np.float64)
        radii = np.asarray(p["radii"], dtype=np.float64)
        rotation = Rotation.from_quat(p["quaternion_wxyz"], scalar_first=True).as_matrix()
        extent = np.sqrt((rotation**2) @ (radii**2))
        shapes.append((p["name"], center, radii, rotation, extent))
    h = args.spacing
    lower = np.floor(np.min([c-e for _,c,_,_,e in shapes], axis=0)/h)*h - 2*h
    upper = np.ceil(np.max([c+e for _,c,_,_,e in shapes], axis=0)/h)*h + 2*h
    axes = [np.linspace(lo, hi, int(round((hi-lo)/h))+1, dtype=np.float32) for lo,hi in zip(lower,upper)]
    grid_shape = tuple(len(axis) for axis in axes)
    xx, yy, zz = np.meshgrid(*axes, indexing="ij", sparse=True)
    field = np.full(grid_shape, np.inf, dtype=np.float32)
    for name, center, radii, rotation, extent in shapes:
        delta = (xx-center[0], yy-center[1], zz-center[2])
        normalized_sq = np.zeros(grid_shape, dtype=np.float32)
        for k in range(3):
            local = sum(delta[j]*rotation[j,k] for j in range(3))
            normalized_sq += np.asarray((local/radii[k])**2, dtype=np.float32)
        # This is a signed implicit field with the correct ellipsoid zero set,
        # not a claim of an exact Euclidean signed-distance function.
        implicit = (np.sqrt(normalized_sq)-1)*float(np.min(radii))
        np.minimum(field, implicit, out=field)
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(h,h,h),
                                          allow_degenerate=False, gradient_direction="ascent")
    vertices = vertices.astype(np.float64)+lower
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals(multibody=True)
    mesh.remove_unreferenced_vertices()
    components = sorted(mesh.split(only_watertight=False), key=lambda p:abs(p.volume), reverse=True)
    # Near a nearly tangent contact, sampled zero sets can enclose tiny internal
    # air pockets. They are not part of the teddy's outer surface. Remove only
    # sub-voxel shells whose every vertex is verified inside the main exterior.
    # Genuine disconnected outside parts are retained and reported, never bridged.
    excluded = []
    kept = [components[0]]
    for component in components[1:]:
        if abs(component.volume) < h**3 and components[0].contains(component.vertices).all():
            excluded.append({"reason":"enclosed sub-voxel interior shell, not outer surface",
                             "volume_m3":float(component.volume), "faces":len(component.faces),
                             "bounds_m":component.bounds.tolist()})
        else:
            kept.append(component)
    mesh = trimesh.util.concatenate(kept)
    components = kept
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError("Union surface is not closed and consistently oriented")
    if any(part.volume <= 0 for part in components):
        raise RuntimeError("Union surface has an inverted or zero-volume component")
    mesh.export(ROOT / "bear_collision_union.obj")
    component_reports = []
    for comp in sorted(components, key=lambda p:p.volume, reverse=True):
        component_reports.append({"vertices":len(comp.vertices), "faces":len(comp.faces),
                                  "watertight":bool(comp.is_watertight), "volume_m3":float(comp.volume),
                                  "bounds_m":comp.bounds.tolist()})
    report = {
        "provenance": "Only the independent four-view reconstruction manifest; no original simulation assets.",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "output_mesh": "bear_collision_union.obj", "coordinate_frame": "world coordinates, metres; no pose offset needed",
        "collision_parts": [name for name,*_ in shapes],
        "method": "Minimum of analytic ellipsoid implicit fields, followed by zero-level marching cubes",
        "field_is_exact_euclidean_sdf": False,
        "spacing_m": h, "grid_shape": list(grid_shape), "grid_bounds_m": [lower.tolist(),upper.tolist()],
        "vertices":len(mesh.vertices), "faces":len(mesh.faces),
        "watertight":bool(mesh.is_watertight), "winding_consistent":bool(mesh.is_winding_consistent),
        "connected_components":len(components), "volume_m3":float(mesh.volume),
        "bounds_m":mesh.bounds.tolist(), "components":component_reports,
        "removed_interior_shells":excluded,
        "geometry_changes": "No manifest dimensions or poses changed; no smoothing, inflation or gap bridging. Outer surface is discretized at the stated grid spacing.",
        "visual_mesh_relationship": "The analytic ellipsoids have the manifest radii; the separate visual parts are faceted icospheres, so their outer surfaces differ slightly.",
        "loading": "Use Mesh with pos=(0,0,0), fixed=True, convexify=False, decimate=False. Disable the separate bear-part collisions when using this union, or internal contact surfaces remain.",
    }
    (ROOT / "bear_collision_union.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key:report[key] for key in ("output_mesh","vertices","faces","watertight","winding_consistent","connected_components","spacing_m")},indent=2))


if __name__ == "__main__":
    main()
