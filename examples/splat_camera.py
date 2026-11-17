"""Camera frames from a Gaussian splat of a real hallway, physics and 2D view from f1tenth_gym.

Exports the roboracer_3dgs_sim scene's track as a gym map (once), wraps the environment with
SplatCameraWrapper and drives two cars with pure pursuit on the scene's raceline (the second one
at 70 % of the raceline speed, starting ahead; the ego at 80 %, keeping its distance). Writes a video of the ego car's camera (20 Hz)
combined with the gym's own 2D renderer:

    --layout side      the camera next to the 2D view following the ego car
    --layout minimap   the camera with the 2D map view as an inset

    xvfb-run -a uv run python examples/splat_camera.py --layout side --seconds 30

The 2D renderer is OpenGL and needs an X display (``xvfb-run`` gives a virtual one). Needs
roboracer_3dgs_sim installed in the same environment (docs/splat_camera.rst).
"""
import argparse
import pathlib

import cv2
import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS
from f1tenth_gym.envs.env_config import EnvConfig, RenderConfig, ResetConfig
from f1tenth_gym.envs.reset import ResetStrategy
from f1tenth_gym.envs.splat import SplatCameraWrapper, export_scene_track

FPS = 20.0


def pure_pursuit(pose, waypoints, lookahead, wheelbase, speed_gain):
    """(steer, speed) towards the raceline point `lookahead` ahead of the nearest one."""
    x, y, yaw = pose
    d = np.hypot(waypoints[:, 0] - x, waypoints[:, 1] - y)
    i = int(np.argmin(d))
    n = len(waypoints)
    j = i
    while np.hypot(waypoints[j % n, 0] - x, waypoints[j % n, 1] - y) < lookahead and j < i + n:
        j += 1
    tx, ty, v = waypoints[j % n]
    local_y = -np.sin(yaw) * (tx - x) + np.cos(yaw) * (ty - y)
    steer = np.arctan(2 * wheelbase * local_y / max(lookahead, 1e-3) ** 2)
    return float(steer), float(v * speed_gain)


def label(img, text, org=(14, 36), scale=0.9):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def compose(camera, view, layout):
    """One BGR video frame from the camera image (RGB) and the gym's 2D frame (RGB)."""
    cam = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
    top = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    h = cam.shape[0]
    if layout == "side":
        top = cv2.resize(top, (h, h), interpolation=cv2.INTER_AREA)
        label(top, "f1tenth_gym 2D", org=(14, h - 18), scale=0.7)
        return np.hstack([cam, top])
    # minimap: the map view cropped to the map itself, in the bottom-right corner
    content = np.argwhere(top.max(axis=2) > 30)
    (r0, c0), (r1, c1) = content.min(axis=0), content.max(axis=0) + 1
    top = top[r0:r1, c0:c1]
    width = cam.shape[1] * 2 // 5
    height = int(round(top.shape[0] * width / top.shape[1]))
    inset = cv2.resize(top, (width, height), interpolation=cv2.INTER_AREA)
    out = cam.copy()
    m = 14
    y0, x0 = h - height - m, cam.shape[1] - width - m
    cv2.rectangle(out, (x0 - 3, y0 - 3), (x0 + width + 2, y0 + height + 2), (255, 255, 255), 3)
    out[y0:y0 + height, x0:x0 + width] = inset
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="hallway_loop_2b")
    ap.add_argument("--layout", choices=["side", "minimap"], default="side")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="video file (default splat_camera_<layout>.mp4)")
    a = ap.parse_args()
    out = a.out or f"splat_camera_{a.layout}.mp4"

    maps = pathlib.Path("maps") / a.scene
    yaml_path = maps / f"{a.scene}.yaml"
    if not yaml_path.exists():
        yaml_path = export_scene_track(a.scene, maps)

    render_cfg = RenderConfig(render_fps=int(FPS), window_size=800,
                              focus_on="agent_0" if a.layout == "side" else None, show_lap_info=False)
    cfg = EnvConfig(map_name=str(yaml_path), num_agents=2, render_config=render_cfg,
                    reset_config=ResetConfig(strategy=ResetStrategy.RL_GRID_STATIC))
    env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="rgb_array")
    env = SplatCameraWrapper(env, scene=a.scene, agents=["agent_0"], camera_hz=FPS)
    track = env.unwrapped.track
    if env.unwrapped.renderer is not None:
        track.raceline.render_waypoints(env.unwrapped.renderer)
    waypoints = np.column_stack([track.raceline.xs, track.raceline.ys, track.raceline.vxs])
    wb = F1TENTH_VEHICLE_PARAMETERS.lf + F1TENTH_VEHICLE_PARAMETERS.lr

    obs, info = env.reset(seed=0)
    first = compose(obs["agent_0"]["image"], env.render(), a.layout)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (first.shape[1], first.shape[0]))
    dt = env.unwrapped.timestep
    every = max(1, int(round(1.0 / FPS / dt)))
    resets, crashes, speeds = 0, 0, []
    for k in range(int(a.seconds / dt)):
        poses = env.unwrapped.sim.state.poses
        action = np.zeros((2, 2), np.float32)
        action[0] = pure_pursuit(poses[0], waypoints, 1.2, wb, 0.8)
        action[1] = pure_pursuit(poses[1], waypoints, 1.2, wb, 0.7)
        # The ego keeps its distance: within 2 m behind the other car it matches its speed,
        # slower only when closer than 1 m.
        gap = poses[1][:2] - poses[0][:2]
        ahead = np.dot(gap, [np.cos(poses[0][2]), np.sin(poses[0][2])]) > 0
        if ahead and np.linalg.norm(gap) < 2.0:
            factor = float(np.clip((np.linalg.norm(gap) - 0.6) / 0.4, 0.0, 1.0))
            action[0, 1] = min(action[0, 1], action[1, 1] * factor)
        obs, reward, terminated, truncated, info = env.step(action)
        speeds.append(float(env.unwrapped.sim.state.standard_state[0, 3]))
        view = env.render()                           # the gym grabs a new 2D frame at render_fps
        if k % every == 0:
            frame = compose(obs["agent_0"]["image"], view, a.layout)
            v = float(env.unwrapped.sim.state.standard_state[0, 3])
            q = info["camera_quality"]["agent_0"]
            label(frame, f"3DGS camera   t {k * dt:5.1f} s   v {v:4.1f} m/s   render quality {q:.2f}")
            writer.write(frame)
        if terminated or truncated:
            resets += 1
            crashes += int(np.any(env.unwrapped.sim.state.collisions > 0))
            obs, info = env.reset(seed=k)
    writer.release()
    env.close()
    print(f"wrote {out} ({resets} episode resets, {crashes} of them collisions; ego mean speed {np.mean(speeds):.1f} m/s, "
          f"max {np.max(speeds):.1f} m/s)")


if __name__ == "__main__":
    main()
