#!/usr/bin/env python3
"""Create an isolated alternate inferred bear, preserving the original assets.

The fuller torso and outward arm placement are a reconstruction hypothesis for
the garment's observed length and sleeve spacing. They are not recovered ground
truth. Existing manifest.json, bear_parts and cloth files remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


ROOT = Path(__file__).resolve().parent


def ellipsoid(path, radii, color):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.vertices *= np.asarray(radii)
    mesh.visual.vertex_colors = np.tile(np.rint(np.asarray(color)*255).astype(np.uint8), (len(mesh.vertices),1))
    mesh.export(path)


def main():
    source_path = ROOT / "manifest.json"
    source_bytes = source_path.read_bytes()
    original = json.loads(source_bytes)
    variant = copy.deepcopy(original)
    part_dir = ROOT / "bear_parts_fitted"
    part_dir.mkdir(exist_ok=True)
    by_name = {p["name"]:p for p in variant["bear"]["parts"]}
    source_parts = {p["name"]:p for p in original["bear"]["parts"]}

    torso = by_name["torso"]
    torso["radii"] = [.120, .103, .145]
    torso["mesh"] = "bear_parts_fitted/torso.obj"
    ellipsoid(ROOT / torso["mesh"], torso["radii"], torso["color_rgba"])

    belly = by_name["belly_patch"]
    belly["position"][1] = -.100
    belly["radii"] = [.078, .012, .100]
    belly["mesh"] = "bear_parts_fitted/belly_patch.obj"
    ellipsoid(ROOT / belly["mesh"], belly["radii"], belly["color_rgba"])

    for side, sign in (("left",-1),("right",1)):
        name = f"arm_{side}"
        part = by_name[name]
        part["position"][0] = sign*.138
        part["mesh"] = f"bear_parts_fitted/{name}.obj"
        # Identical local shape and orientation; only manifest translation changes.
        shutil.copyfile(ROOT / source_parts[name]["mesh"], ROOT / part["mesh"])

    a,b = .120,.103
    h = ((a-b)/(a+b))**2
    perimeter = float(np.pi*(a+b)*(1+3*h/(10+np.sqrt(4-3*h))))
    variant["provenance"] = "Alternate independently inferred geometry based on the same four initial views and garment proportions; not recovered ground truth."
    variant["variant"] = {
        "name": "fuller_torso_shifted_arms",
        "source_manifest": "manifest.json",
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "is_ground_truth": False,
        "changed_parts": ["torso","belly_patch","arm_left","arm_right"],
        "reason": "Explore a fuller torso consistent with the approximate rear-view torso/head width ratio and the garment's 0.70 m length / 0.35 m armhole separation.",
        "equatorial_torso_perimeter_estimate_m": perimeter,
        "assumptions": [
            "Torso horizontal radii are 0.120 m and 0.103 m; center and vertical radius remain unchanged.",
            "Belly patch is widened and moved forward to match the alternate torso.",
            "Each arm is translated 0.015 m outwards; shape and orientation are unchanged.",
            "All remaining mesh paths, poses and colors reference the original independent reconstruction.",
            "Matching an approximate equatorial circumference does not guarantee cloth fit at other heights or task success.",
        ],
    }
    variant["bear"]["validation_arms"] = {
        "left": {"shoulder":[-.105,0,1.025],"tip":[-.165,-.008,.925]},
        "right": {"shoulder":[.105,0,1.025],"tip":[.165,-.008,.925]},
    }
    variant["bear"]["modeling_assumptions"].append("This fitted variant is an explicit alternate geometry hypothesis; see variant metadata.")

    # A correctly posed whole-bear preview avoids pointing the variant manifest
    # at the original GLB. It remains separate from every original output.
    scene = trimesh.Scene()
    for p in variant["bear"]["parts"]:
        mesh = trimesh.load_mesh(ROOT / p["mesh"], process=False)
        mesh.visual.vertex_colors = np.tile(np.rint(np.array(p["color_rgba"])*255).astype(np.uint8), (len(mesh.vertices),1))
        transform = np.eye(4)
        transform[:3,:3] = Rotation.from_quat(p["quaternion_wxyz"], scalar_first=True).as_matrix()
        transform[:3,3] = p["position"]
        scene.add_geometry(mesh, geom_name=p["name"], node_name=p["name"], transform=transform)
    (ROOT / "bear_fitted.glb").write_bytes(scene.export(file_type="glb"))
    variant["bear"]["world_visual_glb"] = "bear_fitted.glb"
    output_path = ROOT / "manifest_fitted.json"
    output_path.write_text(json.dumps(variant,ensure_ascii=False,indent=2)+"\n")
    assert source_path.read_bytes() == source_bytes, "Original manifest changed unexpectedly"
    unchanged = set(by_name)-set(variant["variant"]["changed_parts"])
    assert all(by_name[name] == source_parts[name] for name in unchanged)
    assert variant["cloth"] == original["cloth"]
    print(json.dumps({"manifest":str(output_path),"changed_parts":variant["variant"]["changed_parts"],
                      "equatorial_torso_perimeter_estimate_m":perimeter,
                      "unchanged_part_count":len(unchanged),"original_manifest_preserved":True},indent=2))


if __name__ == "__main__":
    main()
