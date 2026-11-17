"""Splat camera support (f1tenth_gym.envs.splat): track export and the camera wrapper.

The exporter is tested on a synthetic scene bundle (no renderer needed). The wrapper test needs
wgpu (the ``splat`` extra), the hallway_loop_2b scene under $ROBORACER_DATA/scenes and a renderer
device; it is skipped otherwise.
"""
import json
import pathlib
import tempfile
import unittest

import numpy as np

from f1tenth_gym.envs.splat import export_scene_track
from f1tenth_gym.envs.track import Track


def _fake_scene(root: pathlib.Path) -> pathlib.Path:
    """A 10 x 6 m rectangle with a 1 m wide ring corridor, line_centre on the corridor's middle."""
    cell = 0.05
    nx, ny = 200, 120
    origin = np.array([-5.0, -3.0])
    xs = origin[0] + (np.arange(nx) + 0.5) * cell
    ys = origin[1] + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    outer = (np.abs(X) < 4.5) & (np.abs(Y) < 2.5)
    inner = (np.abs(X) < 3.5) & (np.abs(Y) < 1.5)
    corridor = outer & ~inner
    clearance = np.where(corridor, 0.3, 0.0).astype(np.float32)
    t = np.linspace(0, 1, 200, endpoint=False)
    # rectangle loop at |x| = 4, |y| = 2, counter-clockwise
    per = 2 * (8 + 4)
    s = t * per
    pts = []
    for d in s:
        if d < 8:
            pts.append((-4 + d, -2))
        elif d < 12:
            pts.append((4, -2 + (d - 8)))
        elif d < 20:
            pts.append((4 - (d - 12), 2))
        else:
            pts.append((-4, 2 - (d - 20)))
    line = np.array(pts)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "track.npz", clearance=clearance, grid_cell=cell, grid_origin=origin, line_centre=line)
    (root / "scene.json").write_text(json.dumps({"name": "fake", "world_frame": "test", "track": {"file": "track.npz"}}))
    return root


class TestTrackExport(unittest.TestCase):
    def test_export_loads_with_walls_where_the_scene_has_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = _fake_scene(pathlib.Path(tmp) / "scene")
            yaml_path = export_scene_track(scene, pathlib.Path(tmp) / "map", speed=2.0)
            track = Track.from_track_path(yaml_path)
            self.assertAlmostEqual(track.spec.resolution, 0.05)
            np.testing.assert_allclose(track.spec.origin[:2], (-5.0, -3.0))
            occ = track.occupancy_map    # rows = y after the load flip, 255 free, 0 occupied

            def free(x, y):
                c = int((x - track.spec.origin[0]) / track.spec.resolution)
                r = int((y - track.spec.origin[1]) / track.spec.resolution)
                return occ[r, c] > 0

            self.assertTrue(free(0.0, -2.0))      # corridor, bottom
            self.assertTrue(free(4.0, 0.0))       # corridor, right
            self.assertFalse(free(0.0, 0.0))      # the inner block
            self.assertFalse(free(0.0, -2.8))     # outside the outer wall
            np.testing.assert_allclose(track.raceline.vxs, 2.0, atol=1e-5)
            # the reference line goes through the corridor
            d = np.hypot(track.centerline.xs, track.centerline.ys)
            self.assertTrue(np.all((d > 1.9) & (d < 4.5)))


class TestSplatCameraWrapper(unittest.TestCase):
    def setUp(self):
        try:
            import wgpu  # noqa: F401

            from f1tenth_gym.envs.splat import SplatScene
            SplatScene("hallway_loop_2b")
        except Exception as ex:  # no wgpu, or no scene
            self.skipTest(f"wgpu / the hallway_loop_2b scene not available: {ex}")

    def test_images_in_observations(self):
        import gymnasium as gym

        from f1tenth_gym.envs.env_config import EnvConfig
        from f1tenth_gym.envs.splat import SplatCameraWrapper

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = export_scene_track("hallway_loop_2b", tmp)
            env = gym.make("f1tenth_gym:f1tenth-v0",
                           config=EnvConfig(map_name=str(yaml_path), num_agents=2, render_enabled=False))
            try:
                env = SplatCameraWrapper(env, scene="hallway_loop_2b", agents=["agent_0"], camera_hz=20.0)
            except Exception as ex:   # no GPU/CPU adapter for wgpu on this machine
                self.skipTest(f"renderer not available: {ex}")
            obs, info = env.reset(seed=0)
            image = obs["agent_0"]["image"]
            self.assertEqual(image.shape, env.observation_space["agent_0"]["image"].shape)
            self.assertEqual(image.dtype, np.uint8)
            self.assertNotIn("image", obs["agent_1"])
            self.assertGreater(info["camera_quality"]["agent_0"], 0.5)   # the start is on the captured path
            self.assertGreater(float(image.std()), 5.0)                   # not a blank frame
            action = np.array([[0.0, 1.0], [0.0, 1.0]], np.float32)
            obs1, *_ = env.step(action)                                   # 0.01 s later: the cached frame
            np.testing.assert_array_equal(obs1["agent_0"]["image"], image)
            self.assertIsNot(obs1["agent_0"]["image"], image)             # a copy, never a live buffer
            for _ in range(5):
                obs2, *_ = env.step(action)                               # 0.06 s: a new frame
            self.assertGreater(np.abs(obs2["agent_0"]["image"].astype(int) - image).mean(), 0.1)
            env.close()
