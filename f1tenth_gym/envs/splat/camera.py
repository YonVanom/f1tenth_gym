"""Camera observations rendered from a Gaussian splat of a real track.

``SplatCameraWrapper`` wraps an ``F110Env`` (or ``JaxF110Env``) whose map was exported from a
roboracer_3dgs_sim scene bundle with ``export_scene_track`` (same world frame). After every
``reset``/``step`` it takes each camera car's pose from the simulator, places the camera on it and
renders the splat with roboracer_3dgs_sim's WebGPU renderer (the same renderer, camera model and
mount as roboracer_3dgs_sim). The other cars are drawn into the image with a car splat.

Observation: per-agent dictionaries gain ``key`` (default ``"image"``): ``(H, W, 3)`` uint8 RGB,
960 x 600 by default (the RoboRacer's ZED X Mini, rectified). ``info["camera_quality"]`` maps each
camera agent to the scene's render-quality score in [0, 1]: the splat only looks right near the
poses it was captured from (below ``scene.quality_min`` it is noticeably degraded).

Frames: the gym's pose is the centre of gravity; the camera mount (``scene.json``) is relative to
the rear axle, so the rear axle is placed ``lr`` behind the CoG along the heading. The camera height
follows the scene's floor height field under the car (the gym itself is 2D).

The renderer runs on the host, one frame per camera car per step (~15 ms on a desktop GPU), so this
is for perception and imitation learning with Gymnasium environments, not for the device-batched
JAX simulators.
"""
from __future__ import annotations

from typing import Iterable, Optional

import gymnasium as gym
import numpy as np


def _cog_poses_and_lr(env) -> tuple[np.ndarray, float]:
    """(N, 3) CoG poses [x, y, yaw] and the active lr of the wrapped environment."""
    u = env.unwrapped
    core = getattr(u, "core_state", None)
    if core is not None:                                    # JaxF110Env
        model = np.asarray(core.dynamics.model)
        return model[:, [0, 1, 4]].astype(np.float64), float(u.episode_vehicle_params.lr)
    return np.asarray(u.sim.state.poses, np.float64), float(u.sim.vehicle_params.lr)


class SplatCameraWrapper(gym.Wrapper):
    """Adds camera images rendered from a roboracer_3dgs_sim scene to the observations.

    Args:
        env: an f1tenth_gym environment on a map exported from ``scene`` (per-agent observation
            dictionaries; the flat ``DIRECT`` layout is not supported).
        scene: scene bundle folder or name (roboracer_3dgs_sim), e.g. ``"hallway_loop_2b"``.
        agents: agent ids that get a camera (default: all).
        key: observation key of the image.
        draw_cars: draw the other agents into each image with ``car_asset``.
        car_asset: roboracer_3dgs_sim asset name or folder for the other cars.
        device: renderer device: ``"auto"`` (default), ``"gpu"`` or ``"cpu"``.
        camera_hz: render at this rate of simulated time and return the latest frame in between
            (a real camera's frame rate; the gym steps at 100 Hz by default). None: every step.
    """

    def __init__(self, env: gym.Env, scene: str = "hallway_loop_2b", agents: Optional[Iterable[str]] = None,
                 key: str = "image", draw_cars: bool = True, car_asset: str = "car", device: Optional[str] = None,
                 camera_hz: Optional[float] = None):
        super().__init__(env)
        try:
            from roboracer_3dgs_sim.camera import camera_c2w
            from roboracer_3dgs_sim.scene import load_scene
        except ImportError as ex:
            raise ImportError("SplatCameraWrapper needs roboracer_3dgs_sim (see docs/splat_camera.rst)") from ex
        self._camera_c2w = camera_c2w
        self.scene = load_scene(scene)
        self.renderer = self.scene.renderer(device=device, verbose=False)
        self.key = key
        self.camera_period = None if camera_hz is None else 1.0 / float(camera_hz)
        self._next_capture = 0.0
        self._last = None
        n = env.unwrapped.num_agents
        self.agent_ids = [f"agent_{i}" for i in range(n)]
        self.camera_agents = list(agents) if agents is not None else list(self.agent_ids)
        unknown = set(self.camera_agents) - set(self.agent_ids)
        if unknown:
            raise ValueError(f"unknown agents {sorted(unknown)}; the environment has {self.agent_ids}")
        self._car_id = None
        if draw_cars and n > 1:
            from roboracer_3dgs_sim.assets import load_asset
            self._car_id = self.renderer.add_asset(load_asset(car_asset))
        h, w = self.renderer.height, self.renderer.width
        image_space = gym.spaces.Box(0, 255, (h, w, 3), np.uint8)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Dict) or not all(a in space.spaces for a in self.camera_agents):
            raise ValueError("SplatCameraWrapper needs per-agent observation dictionaries (not DIRECT)")
        self.observation_space = gym.spaces.Dict({
            a: (gym.spaces.Dict({**s.spaces, key: image_space}) if a in self.camera_agents else s)
            for a, s in space.spaces.items()})

    # ------------------------------------------------------------------ rendering

    def rear_axle_poses(self) -> np.ndarray:
        """(N, 3) rear axle poses [x, y, yaw] of all cars (the frame the scene's camera mount uses)."""
        cog, lr = _cog_poses_and_lr(self.env)
        c, s = np.cos(cog[:, 2]), np.sin(cog[:, 2])
        return np.column_stack([cog[:, 0] - lr * c, cog[:, 1] - lr * s, cog[:, 2]])

    def render_cameras(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Render every camera agent's image now: ({agent: image}, {agent: render quality})."""
        from roboracer_3dgs_sim.assets import object_pose
        track, mount = self.scene.track, self.scene.mount
        poses = self.rear_axle_poses()
        images, quality = {}, {}
        for agent in self.camera_agents:
            i = self.agent_ids.index(agent)
            c2w = self._camera_c2w(track, *poses[i], mount)
            objects = None
            if self._car_id is not None:
                objects = [(self._car_id, object_pose(track, *poses[j])) for j in range(len(poses)) if j != i]
            images[agent] = self.renderer.render(c2w, objects=objects)
            quality[agent] = float(self.scene.coverage.score(c2w)[0])
        return images, quality

    def _add(self, obs, info, sim_time: float):
        if self._last is None or self.camera_period is None or sim_time >= self._next_capture - 1e-9:
            self._last = self.render_cameras()
            if self.camera_period is not None:
                self._next_capture = max(self._next_capture + self.camera_period, sim_time)
        images, quality = self._last
        obs = dict(obs)
        for agent, image in images.items():
            obs[agent] = {**obs[agent], self.key: image.copy()}   # never hand out a live buffer
        info = dict(info)
        info["camera_quality"] = quality
        return obs, info

    # ------------------------------------------------------------------ gym API

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last, self._next_capture = None, 0.0
        return self._add(obs, info, 0.0)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs, info = self._add(obs, info, float(getattr(self.env.unwrapped, "sim_time", 0.0)))
        return obs, reward, terminated, truncated, info
