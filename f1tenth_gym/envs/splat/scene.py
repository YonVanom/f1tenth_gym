"""Scene bundles and car assets made with roboracer_3dgs_builder, read without any other package.

A **scene bundle** is a folder:

    scene.json   format, splat settings, camera mount, coverage parameters
    splat.bin    the Gaussian splat (layout: float32 position (N,3), scale (N,3), quaternion
                 w x y z (N,4), density (N,), float16 SH (N,48))
    track.npz    the track fitted to the same lidar map: floor height field, drivable-area grid
                 and clearance, closed racelines ``line_<name>`` (counter-clockwise order), and the
                 poses the splat's images were captured from

An **asset** (another car drawn into the images) is a folder with ``asset.json`` and ``splat.bin``
in the same splat layout, its frame at the rear axle (x forward, y left, z up from the floor).

Everything is in the scene's world frame (metres, z up). Names are looked up under
``$ROBORACER_DATA`` (default ``~/roboracer_data``): ``scenes/<name>`` and ``assets/<name>``, then
in the repository's ``splat_data/`` (a git checkout ships the ``hallway_loop_2b`` scene and the
``car`` asset there; they are not part of the wheel).
Ported from roboracer_3dgs_sim (scene.py, track.py, camera.py, assets.py).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .renderer import DEFAULT_INTRINSICS


def data_root() -> pathlib.Path:
    """Where scenes and assets are looked up by name: $ROBORACER_DATA or ~/roboracer_data."""
    return pathlib.Path(os.environ.get("ROBORACER_DATA", "~/roboracer_data")).expanduser()


REPO_DATA = pathlib.Path(__file__).resolve().parents[3] / "splat_data"
"""The scene and car asset shipped with a git checkout (absent from an installed wheel)."""


def _find(name_or_path, kind: str, marker: str) -> pathlib.Path:
    path = pathlib.Path(name_or_path).expanduser()
    if path.name == marker:
        path = path.parent
    if (path / marker).exists():
        return path
    candidates = [data_root() / kind / str(name_or_path), REPO_DATA / kind / str(name_or_path)]
    for candidate in candidates:
        if (candidate / marker).exists():
            return candidate
    tried = ", ".join(str(c) for c in [path, *candidates])
    raise FileNotFoundError(f"no {kind[:-1]} {name_or_path!r} (looked for {marker} in {tried})")


# ----------------------------------------------------------------------------- camera geometry

@dataclass(frozen=True)
class Mount:
    """Camera mount: P (the floor point below the optical centre) relative to the rear axle,
    the optical centre's height above the floor, pitch (positive looks down) and roll."""
    forward: float = 0.27
    left: float = 0.025
    height: float = 0.158
    pitch: float = 0.0
    roll: float = 0.0


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


BODY_FROM_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
"""Columns: the optical x (right), y (down) and z (forward) axes in the car's body frame."""


# ----------------------------------------------------------------------------- scene

class SplatScene:
    """A scene bundle: metadata, floor height, camera poses, render-quality score.

    ``camera_c2w(x, y, yaw)`` gives the camera-to-world 4x4 (OpenCV axes: x right, y down, z
    forward) for a rear axle pose, the car resting on the floor. ``quality(c2w)`` scores how close
    a camera pose is to the poses the splat was captured from (1 = on one; the splat looks right
    only near them, below ``quality_min`` walls visibly smear).
    """

    def __init__(self, name_or_path):
        self.path = _find(name_or_path, "scenes", "scene.json")
        self.meta = json.loads((self.path / "scene.json").read_text())
        if self.meta.get("format") != "roboracer/scene":
            raise ValueError(f"{self.path / 'scene.json'}: not a roboracer scene bundle")
        self.name = self.meta["name"]
        track = np.load(self.path / self.meta["track"]["file"])
        self.floor_origin = np.asarray(track["floor_origin"], np.float64)
        self.floor_cell = float(track["floor_cell"])
        self.floor_heights = np.asarray(track["floor_heights"], np.float64)
        self.capture_xyz = np.asarray(track["centers"], np.float64)
        f = np.asarray(track["forward"], np.float64)
        self._capture_yaw = np.arctan2(f[:, 1], f[:, 0])
        self._tree = cKDTree(self.capture_xyz[:, :2])
        m = self.meta["camera"]["mount"]
        self.mount = Mount(forward=m["forward_m"], left=m["left_m"], height=m["height_m"],
                           pitch=math.radians(m["pitch_deg"]), roll=math.radians(m["roll_deg"]))
        k = self.meta["camera"].get("intrinsics")
        self.intrinsics = tuple(k[c] for c in ("fx", "fy", "cx", "cy")) if k else DEFAULT_INTRINSICS
        c = self.meta.get("coverage", {})
        self._sigma = (c.get("sigma_xy_m", 0.3), c.get("sigma_z_m", 0.05), math.radians(c.get("sigma_yaw_deg", 25.0)))
        self.quality_min = float(c.get("quality_min", 0.45))

    @property
    def splat_path(self) -> pathlib.Path:
        return self.path / self.meta["splat"]["file"]

    def renderer(self, device: Optional[str] = None, verbose: bool = False):
        """A SplatRenderer of this scene (960 x 600, the scene's camera intrinsics)."""
        from .renderer import SplatRenderer
        return SplatRenderer(self.splat_path, self.meta["splat"], intrinsics=self.intrinsics, device=device,
                             verbose=verbose)

    def floor_z(self, xy: np.ndarray) -> np.ndarray:
        """Floor height [m] under ground points xy (N, 2), bilinear in the height field."""
        xy = np.atleast_2d(xy)
        position = (xy - self.floor_origin) / self.floor_cell
        low = np.floor(position).astype(int)
        h = self.floor_heights
        cols = np.clip(low[:, 0], 0, h.shape[1] - 2)
        rows = np.clip(low[:, 1], 0, h.shape[0] - 2)
        tx = np.clip(position[:, 0] - cols, 0, 1)
        ty = np.clip(position[:, 1] - rows, 0, 1)
        return ((1 - ty) * ((1 - tx) * h[rows, cols] + tx * h[rows, cols + 1])
                + ty * ((1 - tx) * h[rows + 1, cols] + tx * h[rows + 1, cols + 1]))

    def camera_c2w(self, x: float, y: float, yaw: float, mount: Optional[Mount] = None) -> np.ndarray:
        """Camera-to-world 4x4 for a rear axle pose; the car's pitch and roll follow the floor."""
        mount = mount or self.mount
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-fwd[1], fwd[0]])
        p = np.array([x, y]) + mount.forward * fwd + mount.left * left
        half = 0.15
        z = self.floor_z(np.array([p, p + half * fwd, p - half * fwd, p + half * left, p - half * left]))
        floor_pitch = math.atan2(z[2] - z[1], 2 * half)
        floor_roll = math.atan2(z[3] - z[4], 2 * half)
        world_from_car = _rot_z(yaw) @ _rot_y(floor_pitch) @ _rot_x(floor_roll)
        car_from_camera = _rot_y(mount.pitch) @ _rot_x(mount.roll)
        c2w = np.eye(4)
        c2w[:3, :3] = world_from_car @ car_from_camera @ BODY_FROM_OPTICAL
        c2w[:3, 3] = np.r_[p, z[0]] + world_from_car @ np.array([0, 0, mount.height])
        return c2w

    def object_pose(self, x: float, y: float, yaw: float) -> np.ndarray:
        """4x4 world_from_asset of an object (e.g. a car at its rear axle) standing level on the floor."""
        c, s = math.cos(yaw), math.sin(yaw)
        pose = np.eye(4)
        pose[:2, :2] = [[c, -s], [s, c]]
        pose[:3, 3] = (x, y, float(self.floor_z(np.array([x, y], float))[0]))
        return pose

    def quality(self, c2w: np.ndarray) -> float:
        """Render quality in [0, 1]: a Gaussian falloff in distance, height and yaw to the nearest
        capture pose (a heuristic chosen by eye, not a measurement)."""
        position, forward = c2w[:3, 3], c2w[:3, 2]
        yaw = math.atan2(forward[1], forward[0])
        near = self._tree.query_ball_point(position[:2], 3 * self._sigma[0])
        if not near:
            return 0.0
        near = np.array(near)
        d_xy = np.linalg.norm(self.capture_xyz[near, :2] - position[:2], axis=1)
        d_z = self.capture_xyz[near, 2] - position[2]
        d_yaw = (self._capture_yaw[near] - yaw + np.pi) % (2 * np.pi) - np.pi
        q = np.exp(-0.5 * ((d_xy / self._sigma[0]) ** 2 + (d_z / self._sigma[1]) ** 2 + (d_yaw / self._sigma[2]) ** 2))
        return float(q.max())

    def __repr__(self) -> str:
        return f"SplatScene({self.name!r}, {self.path})"


# ----------------------------------------------------------------------------- assets

@dataclass
class SplatAsset:
    """An object to draw into the images (e.g. a car), its frame at the rear axle on the floor."""
    name: str
    path: pathlib.Path
    splat_path: pathlib.Path
    splat_meta: dict


def load_asset(name_or_path) -> SplatAsset:
    """An asset folder (asset.json + splat.bin) by path, or by name under $ROBORACER_DATA/assets."""
    path = _find(name_or_path, "assets", "asset.json")
    meta = json.loads((path / "asset.json").read_text())
    if meta.get("format") != "roboracer/asset":
        raise ValueError(f"{path / 'asset.json'}: not a roboracer asset")
    splat_path = path / meta["splat"]["file"]
    expected = int(meta["splat"]["count"]) * 140
    if splat_path.stat().st_size != expected:
        raise ValueError(f"{splat_path}: {splat_path.stat().st_size} bytes, expected {expected}")
    return SplatAsset(name=meta["name"], path=path, splat_path=splat_path, splat_meta=meta["splat"])
