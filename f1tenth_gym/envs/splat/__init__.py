"""Camera frames from a Gaussian splat of a real track (scene bundles from roboracer_3dgs_builder).

``export_scene_track`` writes a scene bundle's track as a gym map in the scene's world frame;
``SplatCameraWrapper`` renders each car's camera from the splat after every step, with the WebGPU
renderer in ``renderer.py`` (needs ``wgpu``, imported only when rendering). See docs/splat_camera.rst.
"""
from .scene import SplatScene, load_asset
from .track_export import export_scene_track, speed_profile

__all__ = ["export_scene_track", "speed_profile", "SplatScene", "load_asset", "SplatCameraWrapper", "SplatRenderer"]


def __getattr__(name):
    if name == "SplatCameraWrapper":
        from .camera import SplatCameraWrapper
        return SplatCameraWrapper
    if name == "SplatRenderer":
        from .renderer import SplatRenderer
        return SplatRenderer
    raise AttributeError(name)
