#!/usr/bin/env python3
"""Build independent visual approximations from the four supplied initial views.

No original simulation assets, trajectories, or physical parameters are used.
Run with a Python environment containing numpy, trimesh and Pillow.
All dimensions use metres. Front is -Y; up is +Z; table top is Z=0.72.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh


OUT = Path(__file__).resolve().parent
BROWN = (0.43, 0.245, 0.105, 1.0)
BROWN_LIGHT = (0.53, 0.325, 0.155, 1.0)
TAN = (0.69, 0.60, 0.40, 1.0)
DARK = (0.055, 0.041, 0.025, 1.0)


def rotation_y(theta: float) -> tuple[np.ndarray, list[float]]:
    c, s = np.cos(theta), np.sin(theta)
    mat = np.eye(4)
    mat[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return mat, [float(np.cos(theta / 2)), 0.0, float(np.sin(theta / 2)), 0.0]


def make_bear() -> dict:
    """Export each part at local origin and a colored world-coordinate GLB."""
    part_dir = OUT / "bear_parts"
    part_dir.mkdir(exist_ok=True)
    scene = trimesh.Scene()
    parts = []

    def add(name, center, radii, color=BROWN, collision=True, theta=0.0):
        # Modest faceting intentionally matches the supplied rendering style.
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.vertices *= np.asarray(radii)
        rgba = np.rint(np.asarray(color) * 255).astype(np.uint8)
        mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
        mesh.export(part_dir / f"{name}.obj")
        transform, quat = rotation_y(theta)
        transform[:3, 3] = center
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=transform)
        parts.append({
            "name": name,
            "mesh": f"bear_parts/{name}.obj",
            "position": list(center),
            "quaternion_wxyz": quat,
            "radii": list(radii),
            "color_rgba": list(color),
            "collision": bool(collision),
            "fixed": True,
            "geometry": "ellipsoid",
        })

    add("torso", (0, 0, .91), (.105, .083, .145))
    add("head", (0, -.004, 1.115), (.095, .080, .087), BROWN_LIGHT)
    for side, sign in (("left", -1), ("right", 1)):
        add(f"ear_{side}", (sign * .078, -.002, 1.185), (.034, .026, .034), BROWN_LIGHT)
        add(f"ear_inner_{side}", (sign * .078, -.024, 1.185), (.022, .004, .023), TAN, False)
        add(f"foot_{side}", (sign * .058, -.027, .757), (.046, .060, .033), BROWN_LIGHT)
        add(f"arm_{side}", (sign * .123, -.008, .978), (.026, .026, .069), BROWN_LIGHT,
            True, -sign * np.arctan(.6))
        add(f"eye_{side}", (sign * .031, -.076, 1.132), (.0058, .0042, .007), DARK, False)
        add(f"eye_glint_{side}", (sign * .0305 - .001, -.0797, 1.134),
            (.0015, .0008, .0017), (.94, .90, .79, 1), False)
    add("belly_patch", (0, -.079, .915), (.069, .011, .100), TAN, False)
    add("muzzle", (0, -.079, 1.088), (.043, .025, .032), TAN, False)
    add("nose", (0, -.103, 1.101), (.0092, .006, .0068), DARK, False)
    # The mouth is decorative only. A small tapered stem and two curves are
    # combined into one mesh, avoiding any hidden constraint/collision role.
    points = [np.array([0., -.1045, 1.096]), np.array([0., -.1045, 1.086])]
    mouth_segments = [(points[0], points[1])]
    for sign in (-1, 1):
        curve = np.array([[0, -.1045, 1.086], [sign*.004, -.104, 1.083],
                          [sign*.009, -.103, 1.0835], [sign*.013, -.101, 1.087]])
        mouth_segments.extend(zip(curve[:-1], curve[1:]))
    mouth_parts = []
    mouth_origin = np.array([0., -.103, 1.088])
    for p, q in mouth_segments:
        mouth_parts.append(trimesh.creation.cylinder(radius=.00105, segment=np.array([p, q])-mouth_origin,
                                                     sections=6))
    mouth = trimesh.util.concatenate(mouth_parts)
    mouth.visual.vertex_colors = np.tile(np.rint(np.array(DARK)*255).astype(np.uint8), (len(mouth.vertices), 1))
    mouth.export(part_dir / "mouth.obj")
    transform = np.eye(4)
    transform[:3, 3] = mouth_origin
    scene.add_geometry(mouth, node_name="mouth", geom_name="mouth", transform=transform)
    parts.append({"name": "mouth", "mesh": "bear_parts/mouth.obj", "position": mouth_origin.tolist(),
                  "quaternion_wxyz": [1, 0, 0, 0], "color_rgba": list(DARK), "collision": False,
                  "fixed": True, "geometry": "decorative_tube_mesh"})
    (OUT / "bear.glb").write_bytes(scene.export(file_type="glb"))
    return {"parts": parts, "world_visual_glb": "bear.glb", "part_mesh_frame": "local, centred on part position",
            "front_axis": "-Y", "table_top_z": .72,
            "modeling_assumptions": [
                "Absolute scale is not identifiable from the four uncalibrated images; all dimensions are estimates.",
                "The bear is approximated by fixed overlapping ellipsoids; no plush material parameters are inferred.",
                "Arms are separate elongated ellipsoids angled outward and downward; no bear articulation is inferred.",
                "Small colored facial and belly parts are visual only, with collision disabled.",
            ]}


def segmented_axis(breaks, target=.009):
    values = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(round((b-a)/target)))
        values.extend(np.linspace(a, b, n+1)[:-1])
    return np.array(values+[breaks[-1]], dtype=np.float64)


def in_rounded_hole(x, y, cx, cy=.020, w=.060, h=.064, r=.016):
    d = np.array([abs(x-cx), abs(y-cy)]) - np.array([w/2-r, h/2-r])
    return np.linalg.norm(np.maximum(d, 0)) <= r


def cloth_texture():
    width, height = 2048, 512
    rng = np.random.default_rng(20261020)
    base = np.empty((height, width, 3), dtype=np.float32)
    base[:] = [188, 184, 159]
    noise = rng.normal(0, 1.2, (height, width, 1))
    weave = ((np.arange(width)[None, :, None] % 4 == 0)*1.3
             + (np.arange(height)[:, None, None] % 4 == 0)*1.3)
    base += noise + weave
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    strip = round(width*.028/.70)
    draw.rectangle([0, 0, strip, height], fill=(37, 69, 74))
    draw.rectangle([width-strip, 0, width, height], fill=(130, 83, 45))
    for xx in list(range(4, strip, 7))+list(range(width-strip+4, width, 7)):
        right = xx > width/2
        for yy in range(3, height, 9):
            draw.line((xx, yy, xx, yy+4), fill=(164, 117, 72) if right else (68, 104, 109), width=1)
    # Small dashes are actual UV texture markings, not geometric edge constraints.
    for yy in (7, height-8):
        for xx in range(strip+7, width-strip-7, 17):
            draw.line((xx, yy, xx+8, yy), fill=(127, 128, 111), width=2)
    for xx in (strip+5, width-strip-6):
        for yy in range(12, height-12, 16):
            draw.line((xx, yy, xx, yy+7), fill=(139, 135, 114), width=2)
    # Visual seam around each opening; transparent pixels are unnecessary because
    # the arm openings are absent mesh faces, not alpha-painted fake holes.
    for cx in (-.175, .175):
        def uvpx(x, y):
            return ((x+.35)/.70*width, (.08-y)/.16*height)
        x0,y1 = uvpx(cx-.033, .020-.035)
        x1,y0 = uvpx(cx+.033, .020+.035)
        draw.rounded_rectangle([x0,y0,x1,y1], radius=28, outline=(146,143,123), width=2)
    image.save(OUT / "cloth_texture.png")


def make_cloth() -> dict:
    # Segment boundaries exactly include intended hole extents; grid pitches
    # remain between 8.5 and 9.4 mm rather than introducing tiny sliver faces.
    xs = segmented_axis([-.35, -.205, -.145, .145, .205, .35])
    ys = segmented_axis([-.08, -.012, .052, .08])
    vertices = np.array([[x, y, 0.] for y in ys for x in xs])
    nx = len(xs)
    faces = []
    for j in range(len(ys)-1):
        for i in range(len(xs)-1):
            x, y = (xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2
            if any(in_rounded_hole(x,y,cx) for cx in (-.175,.175)):
                continue
            a, b, c, d = j*nx+i, j*nx+i+1, (j+1)*nx+i+1, (j+1)*nx+i
            faces.extend(((a,b,c),(a,c,d)) if (i+j)%2 == 0 else ((a,b,d),(b,c,d)))
    faces = np.asarray(faces, dtype=np.int64)
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    vertices = vertices[used]
    faces = inverse.reshape(-1,3)
    uv = np.column_stack(((vertices[:,0]+.35)/.70,(vertices[:,1]+.08)/.16))
    # Write explicitly so face and UV indices match vertex indices exactly.
    lines = ["# Independent reconstruction: single layer cloth with two real holes", "mtllib cloth.mtl", "o garment", "usemtl beige_fabric"]
    lines += ["v %.9f %.9f %.9f" % tuple(v) for v in vertices]
    lines += ["vt %.9f %.9f" % tuple(t) for t in uv]
    lines += ["f "+" ".join(f"{k+1}/{k+1}" for k in face) for face in faces]
    (OUT/"cloth.obj").write_text("\n".join(lines)+"\n")
    (OUT/"cloth.mtl").write_text("newmtl beige_fabric\nKa 0.2 0.2 0.2\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd cloth_texture.png\n")
    cloth_texture()
    edges = np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
    unique_edges, counts = np.unique(edges,axis=0,return_counts=True)
    boundary = unique_edges[counts == 1]
    # Count disjoint boundary loops and store explicit node groups for grasps.
    adj = {}
    for a,b in boundary:
        adj.setdefault(int(a),[]).append(int(b)); adj.setdefault(int(b),[]).append(int(a))
    assert all(len(v)==2 for v in adj.values()), "Non-manifold cloth boundary"
    unvisited = set(adj)
    loops = []
    while unvisited:
        start = min(unvisited); prev = None; node = start; loop = []
        while node not in loop:
            loop.append(node); unvisited.discard(node)
            nxt = next(n for n in adj[node] if n != prev)
            prev, node = node, nxt
        loops.append(loop)
    loops.sort(key=len,reverse=True)
    assert len(loops) == 3, f"Expected perimeter and two armholes, got {len(loops)}"
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    assert len(mesh.split(only_watertight=False)) == 1
    assert mesh.euler_number == -1
    np.savez(OUT/"cloth_topology.npz", vertices=vertices, faces=faces, uv=uv,
             left_velcro=np.flatnonzero(vertices[:,0] <= -.35+.028+1e-9),
             right_velcro=np.flatnonzero(vertices[:,0] >= .35-.028-1e-9),
             left_edge=np.flatnonzero(np.isclose(vertices[:,0],-.35)),
             right_edge=np.flatnonzero(np.isclose(vertices[:,0],.35)))
    hole_loops = sorted(loops[1:],key=lambda ids: vertices[ids,0].mean())
    return {
        "mesh": "cloth.obj", "material": "cloth.mtl", "texture": "cloth_texture.png",
        "topology": "cloth_topology.npz", "frame": "XY plane, centered at origin, normal +Z",
        "length_x": .70, "width_y": .16, "thickness": None,
        "vertex_count": len(vertices), "triangle_count": len(faces),
        "grid_spacing_x_range": [float(np.diff(xs).min()),float(np.diff(xs).max())],
        "grid_spacing_y_range": [float(np.diff(ys).min()),float(np.diff(ys).max())],
        "boundary_loop_count": len(loops), "euler_characteristic": int(mesh.euler_number),
        "outer_boundary_indices": loops[0],
        "holes": [{"center_xy": [cx,.020], "target_width_x": .060, "target_height_y": .064,
                   "target_corner_radius": .016, "boundary_vertex_indices": loop,
                   "actual_bounds_xy": [vertices[loop,:2].min(axis=0).tolist(),vertices[loop,:2].max(axis=0).tolist()]}
                  for cx,loop in zip((-.175,.175),hole_loops)],
        "velcro": {"strip_width_x": .028, "left_color_rgba": [37/255,69/255,74/255,1],
                   "right_color_rgba": [130/255,83/255,45/255,1],
                   "physical_adhesion": "Not encoded in asset. Requires a separate explicitly stated simulation model."},
        "modeling_assumptions": [
            "The garment is a single rectangular sheet with two cut-out armholes. No unsupported sleeve tubes are added.",
            "The rounded rectangular openings use a staircase grid boundary and absent mesh faces.",
            "Material stiffness, density, friction, thickness and adhesion cannot be recovered from still images.",
            "Every cloth dimension and the hidden-facing appearance are independent reconstruction assumptions.",
        ],
    }


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {"provenance": "Independent procedural reconstruction from four supplied initial-state images and task description only.",
                "units": "metres", "world_axes": {"up": "+Z", "bear_front": "-Y"},
                "bear": make_bear(), "cloth": make_cloth()}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"manifest": str(OUT/"manifest.json"), "bear_parts": len(manifest["bear"]["parts"]),
                      "cloth_vertices": manifest["cloth"]["vertex_count"],
                      "cloth_triangles": manifest["cloth"]["triangle_count"],
                      "boundary_loops": manifest["cloth"]["boundary_loop_count"]},indent=2))


if __name__ == "__main__":
    main()
