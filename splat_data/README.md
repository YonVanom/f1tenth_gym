# Splat data for the camera wrapper

Data for `f1tenth_gym.envs.splat` (docs/splat_camera.rst), found by name from a git checkout:

| folder | what |
|---|---|
| `scenes/hallway_loop_2b/` | a real hallway loop (62 m, both directions): Gaussian splat (`splat.bin`, 405,366 particles), the track fitted to the same lidar map (`track.npz`: floor, drivable area, racelines, capture poses), `scene.json` |
| `assets/car/` | the RoboRacer car, scanned (41,906 particles), for drawing the other cars into camera frames |

Recorded with a RoboRacer car (ZED X Mini camera, Livox Mid-360 lidar) and built with
roboracer_3dgs_builder. The files are not part of the installed package: from a wheel, put them
under `$ROBORACER_DATA/scenes/` and `$ROBORACER_DATA/assets/`, or pass their paths.
