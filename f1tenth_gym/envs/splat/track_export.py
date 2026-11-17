"""Export a scene bundle's track as an f1tenth_gym map.

The scene bundle (``scene.json`` + ``splat.bin`` + ``track.npz``, made with
roboracer_3dgs_builder, see scene.py) holds a Gaussian splat of a real
track and the track geometry fitted to the same lidar map: the drivable area on a grid, the floor
height and closed racelines, all in the scene's world frame (metres, z up). The gym needs a
ROS-style occupancy map and reference lines. This writes them in the *same* world frame, so the
gym's physics walls are where the splat shows walls and a gym pose can be rendered directly:

    <out>/<name>.png               occupancy image: drivable floor white (free), the rest black
    <out>/<name>.yaml              resolution and origin of that image
    <out>/<name>_centerline.csv    x_m, y_m, w_tr_right_m, w_tr_left_m (the scene's ``centre`` line)
    <out>/<name>_raceline.csv      s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2

The raceline speeds are a simple profile (lateral-acceleration limit in the bends, braking and
accelerating limits between them) or a constant ``speed``. Load the result with
``EnvConfig(map_name="<out>/<name>.yaml")``.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np
from PIL import Image

from ..track.cubic_spline import CubicSplineND


def _scene_files(scene: str | pathlib.Path) -> tuple[pathlib.Path, dict]:
    from .scene import _find
    path = _find(scene, "scenes", "scene.json")
    return path, json.loads((path / "scene.json").read_text())


def speed_profile(ss: np.ndarray, ks: np.ndarray, v_max: float = 8.0, a_lat: float = 3.0,
                  a_accel: float = 2.5, a_brake: float = 4.0) -> np.ndarray:
    """Speeds along a closed line: at most sqrt(a_lat / |curvature|) and v_max, reachable with the
    acceleration and braking limits (forward and backward passes, twice round the loop)."""
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(ks), 1e-6)))
    ds = np.diff(np.r_[ss, ss[-1] + (ss[-1] - ss[-2])])
    n = len(v)
    for _ in range(2):
        for i in range(2 * n):                           # accelerating
            a, b = i % n, (i + 1) % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * a_accel * ds[a]))
        for i in range(2 * n, 0, -1):                    # braking
            a, b = (i - 1) % n, i % n
            v[a] = min(v[a], np.sqrt(v[b] ** 2 + 2 * a_brake * ds[a]))
    return v


def export_scene_track(scene: str | pathlib.Path, out_dir: str | pathlib.Path, *, line: str = "centre",
                       name: Optional[str] = None, speed: Optional[float] = None) -> pathlib.Path:
    """Write the gym map files for a scene bundle; returns the path of the map YAML.

    Args:
        scene: a scene bundle folder (or its scene.json), or a scene name under
            ``$ROBORACER_DATA/scenes`` (e.g. ``"hallway_loop_2b"``).
        out_dir: folder to write into (created).
        line: the scene's raceline used for the gym's raceline (``centre``, ``mincurv``, ...);
            the centerline is always the scene's ``centre``.
        name: map name (default: the scene's name).
        speed: constant raceline speed [m/s]; default: ``speed_profile`` of the line.
    """
    path, meta = _scene_files(scene)
    name = name or meta["name"]
    data = np.load(path / meta["track"]["file"])
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Occupancy: rows of the grid are y (up), the gym flips the image vertically on load.
    free = data["clearance"] > 0
    image = np.where(free, 254, 0).astype(np.uint8)
    Image.fromarray(np.flipud(image)).save(out / f"{name}.png")
    cell = float(data["grid_cell"])
    ox, oy = (float(v) for v in data["grid_origin"])
    (out / f"{name}.yaml").write_text(
        f"image: {name}.png\nresolution: {cell}\norigin: [{ox}, {oy}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
        f"# exported from the scene {meta['name']!r}; world frame: {meta.get('world_frame', '')}\n")

    def lookup_clearance(xy: np.ndarray) -> np.ndarray:
        ij = np.floor((xy - np.array([ox, oy])) / cell).astype(int)
        rows = np.clip(ij[:, 1], 0, data["clearance"].shape[0] - 1)
        cols = np.clip(ij[:, 0], 0, data["clearance"].shape[1] - 1)
        return data["clearance"][rows, cols].astype(np.float64)

    centre = np.asarray(data["line_centre"], np.float64)
    width = lookup_clearance(centre)          # distance to the nearest wall: a symmetric estimate
    np.savetxt(out / f"{name}_centerline.csv", np.column_stack([centre, width, width]), delimiter=",",
               fmt="%.4f", header="x_m, y_m, w_tr_right_m, w_tr_left_m")

    points = np.asarray(data[f"line_{line}"], np.float64)
    spline = CubicSplineND(x=points[:, 0], y=points[:, 1], vxs=np.ones(len(points)))
    ss, ks = np.asarray(spline.ss, np.float64), np.asarray(spline.ks, np.float64)
    vx = np.full(len(ss), float(speed)) if speed is not None else speed_profile(ss, ks)
    ax = vx * np.gradient(vx, ss)
    rows = np.column_stack([ss, spline.xs, spline.ys, spline.psis, ks, vx, ax])
    with open(out / f"{name}_raceline.csv", "w") as fh:
        fh.write(f"# raceline {line!r} of the scene {meta['name']!r}\n#\n")
        fh.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
        np.savetxt(fh, rows, delimiter="; ", fmt="%.6f")
    return out / f"{name}.yaml"
