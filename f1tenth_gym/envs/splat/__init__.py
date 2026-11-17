"""Camera frames from a Gaussian splat of a real track (roboracer_3dgs_sim scenes).

``export_scene_track`` writes a scene bundle's track as a gym map in the scene's world frame;
``SplatCameraWrapper`` renders each car's camera from the splat after every step. The renderer is
roboracer_3dgs_sim's (optional dependency, imported only by the wrapper). See docs/splat_camera.rst.
"""
from .track_export import export_scene_track

__all__ = ["export_scene_track", "SplatCameraWrapper"]


def __getattr__(name):
    if name == "SplatCameraWrapper":
        from .camera import SplatCameraWrapper
        return SplatCameraWrapper
    raise AttributeError(name)
