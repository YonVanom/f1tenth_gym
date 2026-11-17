Camera frames from a Gaussian splat
===================================

.. seealso::

   ``f1tenth_gym.envs.splat`` — ``export_scene_track`` and ``SplatCameraWrapper``;
   ``examples/splat_camera.py`` — a complete run.

The gym simulates the physics in 2D and has a top-down renderer, but no camera. This
adds one: camera frames rendered from a 3D Gaussian splat of a real track, so
perception code can be developed and tested in the gym. A *scene bundle* (made with
roboracer_3dgs_builder from a recording of a car's camera and lidar) is a splat of a
real place plus the track fitted to the same lidar map, in one world frame. The gym
keeps doing all of the physics, LiDAR and collisions; the splat only provides the
pictures. The renderer is part of this package (``f1tenth_gym.envs.splat.renderer``,
WebGPU); it reproduces 3DGUT, the splat's training renderer.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Piece
     - Does
   * - ``export_scene_track(scene, out_dir)``
     - Writes the scene's track as a gym map in the scene's world frame: the
       occupancy image (drivable floor free, the rest occupied), its YAML,
       ``<name>_centerline.csv`` and ``<name>_raceline.csv`` (speeds from
       ``speed_profile``: lateral, braking and acceleration limits).
   * - ``SplatCameraWrapper(env, scene)``
     - After every ``reset``/``step``, renders each camera car's view from the splat
       and adds it to that car's observation as ``"image"`` (960 x 600 RGB,
       ``uint8``). Other cars are drawn with a car splat.
       ``info["camera_quality"]`` gives the render-quality score per camera.

Install
-------

The renderer needs ``wgpu`` (WebGPU: Vulkan, Metal, DirectX 12 or a CPU fallback),
in the optional ``splat`` extra:

.. code-block:: bash

   uv sync --extra splat

A git checkout includes one scene, ``hallway_loop_2b`` (a real 62 m hallway loop,
recorded in both directions), and the scanned RoboRacer as the ``car`` asset, in
``splat_data/``; they are found by name. Other scenes and assets are folders: pass
their paths, or put them under ``$ROBORACER_DATA`` (default ``~/roboracer_data``) as
``scenes/<name>`` and ``assets/<name>``, which is searched first. The data is not part of
the installed wheel. Without a car asset the other cars are not drawn.

Use
---

.. code-block:: python

   import gymnasium as gym
   from f1tenth_gym.envs.env_config import EnvConfig
   from f1tenth_gym.envs.splat import SplatCameraWrapper, export_scene_track

   yaml_path = export_scene_track("hallway_loop_2b", "maps/hallway_loop_2b")
   cfg = EnvConfig(map_name=str(yaml_path))
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   env = SplatCameraWrapper(env, scene="hallway_loop_2b", camera_hz=20.0)
   obs, info = env.reset(seed=0)
   image = obs["agent_0"]["image"]          # (600, 960, 3) uint8 RGB

``camera_hz`` renders at a real camera's rate (simulated time) and returns the latest
frame in between; the gym steps at 100 Hz by default. ``agents=[...]`` limits the
cameras to some cars.

Run the example (two cars, the ego camera next to a top view, written to
``splat_camera.mp4``):

.. code-block:: bash

   uv run python examples/splat_camera.py --seconds 30

Frames and the camera
---------------------

The exported map shares the scene's world frame, so the gym's poses are used as they
are. The scene's camera mount uses a different reference point on the car than the
gym's pose (:doc:`dynamics`); the wrapper converts with the active vehicle's ``lr``.
The gym is 2D: the camera height and the car's pitch and roll follow the scene's floor
height field. Images use the scene's camera model (rectified pinhole, the RoboRacer's
ZED X Mini at 960 x 600, 0.158 m above the floor).

Limits
------

- **Host-side, one frame per camera.** About 15 ms per frame on a desktop GPU. Use it
  with the Gymnasium environments (``F110Env``; ``JaxF110Env`` reads its poses from
  the device state), not inside the device-batched JAX simulators.
- **Near the captured path.** A splat looks right near the poses its images were taken
  from. Far off them (spins, the wrong way round a one-direction capture, the middle
  of a wide room) the frames degrade; ``info["camera_quality"]`` below the scene's
  ``quality_min`` (0.45) flags that.
- **Physics are the gym's.** Vehicle parameters, delays and contact are the gym's own;
  the camera sees the simulated state at the step it was rendered.
