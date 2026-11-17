"""Gaussian-splat renderer on WebGPU (wgpu-py): Vulkan, Metal, DirectX 12, or a CPU fallback.

Ported from roboracer_3dgs_sim (renderer.py) so that f1tenth_gym needs only ``wgpu``. It draws a
scene's 3D Gaussians the way 3DGUT (the splat's training renderer) does: per-particle spherical
harmonics colour, the same culling and screen extents, a per-pixel response evaluated along the
camera ray in the Gaussian's own frame, a global back-to-front order, and front-to-back "over"
blending into a float16 image. It matches 3DGUT's own renders at ~39 dB PSNR.

Per frame:
1. CPU (numpy): depth of every particle centre, drop those closer than 0.2 m or behind the
   camera, sort the rest far to near.
2. GPU: one instanced quad per particle. The vertex shader projects the Gaussian, computes its
   colour and its screen extent; the fragment shader evaluates the Gaussian along the pixel's
   ray and blends premultiplied colour (``one, one-minus-src-alpha``).
3. Read the image back, clamp to [0, 1], round to uint8.

Objects (other cars): ``add_asset`` appends an asset's Gaussians, in the asset frame, to the
particle buffers once. ``render(c2w, objects=[(asset_id, world_from_asset), ...])`` draws them
with one rigid transform per instance, depth-sorted together with the scene's particles.

Device: ``device='auto'`` (default) takes the best GPU on a native backend and falls back to a
software (CPU) adapter; ``'gpu'`` refuses the fallback, ``'cpu'`` forces it. The environment
variable ``F1TENTH_SPLAT_DEVICE`` sets the default.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_INTRINSICS = (359.4684753417969, 359.4684753417969, 479.6768493652344, 310.8428955078125)
"""fx, fy, cx, cy at 960 x 600: the rectified left eye of the RoboRacer's ZED X Mini."""

MIN_DEPTH = 0.2
"""Particles whose centre is closer to the camera than this [m] are not drawn (as 3DGUT)."""

WGSL = """
struct Cam {
  r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>,   // rows of R_world->cam, .w = t_world->cam
  cam_pos: vec4<f32>,                            // camera centre in the world
  k: vec4<f32>,                                  // fx fy cx cy
  res_minalpha: vec4<f32>,                       // width height min_alpha min_response
  misc: vec4<f32>,                               // max_alpha sh_degree
  obj: vec4<u32>,                                // scene particle count, object instance count
};
// An object instance: world_from_asset rigid transform and where its particles live.
struct Inst {
  r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>,   // rows of R_asset->world, .w = t_asset->world
  idx: vec4<u32>,                                // first order value of the instance, first particle of its asset
};
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> pd: array<vec4<f32>>;     // position, density
@group(0) @binding(2) var<storage, read> rot: array<vec4<f32>>;    // unit quaternion w x y z
@group(0) @binding(3) var<storage, read> scl: array<vec4<f32>>;    // scale (x y z, 0)
@group(0) @binding(4) var<storage, read> sh: array<u32>;           // 48 float16 per particle, 2 per word
@group(0) @binding(5) var<storage, read> order: array<u32>;        // order value, far to near (see vs)
@group(0) @binding(6) var<storage, read> insts: array<Inst>;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) oc: vec3<f32>,   // camera centre in the Gaussian's unit frame
  @location(1) dir: vec3<f32>,                     // pixel ray direction in that frame
  @location(2) @interpolate(flat) color: vec4<f32>,
};

// SH coefficient j (0..47) of the particle whose coefficients start at half-word b.
fn shv(j: u32) -> f32 {
  let w = unpack2x16float(sh[j >> 1u]);
  return select(w.x, w.y, (j & 1u) == 1u);
}
fn shc(b: u32, k: u32) -> vec3<f32> {
  return vec3<f32>(shv(b + 3u * k), shv(b + 3u * k + 1u), shv(b + 3u * k + 2u));
}

// Colour of particle i seen along unit direction d (camera to particle, in the particle's frame).
fn sh_color(i: u32, d: vec3<f32>) -> vec3<f32> {
  let b = i * 48u;
  let deg = i32(cam.misc.y);
  var f = 0.28209479177387814 * shc(b, 0u);
  if (deg > 0) {
    let x = d.x; let y = d.y; let z = d.z;
    let C1 = 0.4886025119029199;
    f = f - C1 * y * shc(b, 1u) + C1 * z * shc(b, 2u) - C1 * x * shc(b, 3u);
    if (deg > 1) {
      let xx = x * x; let yy = y * y; let zz = z * z; let xy = x * y; let yz = y * z; let xz = x * z;
      f = f + 1.0925484305920792 * xy * shc(b, 4u) - 1.0925484305920792 * yz * shc(b, 5u)
            + 0.31539156525252005 * (2.0 * zz - xx - yy) * shc(b, 6u)
            - 1.0925484305920792 * xz * shc(b, 7u) + 0.5462742152960396 * (xx - yy) * shc(b, 8u);
      if (deg > 2) {
        f = f - 0.5900435899266435 * y * (3.0 * xx - yy) * shc(b, 9u)
              + 2.890611442640554 * xy * z * shc(b, 10u)
              - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * shc(b, 11u)
              + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shc(b, 12u)
              - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * shc(b, 13u)
              + 1.445305721320277 * z * (xx - yy) * shc(b, 14u)
              - 0.5900435899266435 * x * (xx - 3.0 * yy) * shc(b, 15u);
      }
    }
  }
  return max(f + 0.5, vec3<f32>(0.0));
}

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) inst: u32) -> VOut {
  var out: VOut;
  out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);   // outside the clip volume = not drawn
  out.oc = vec3<f32>(0.0); out.dir = vec3<f32>(0.0); out.color = vec4<f32>(0.0);
  // Order value g < scene count: scene particle g. Otherwise particle g - start + base of the
  // object instance k whose order values start at insts[k].idx.x (ascending over k).
  let g = order[inst];
  var i = g;
  let is_obj = g >= cam.obj.x;
  var Ri = mat3x3<f32>();
  var ti = vec3<f32>(0.0);
  if (is_obj) {
    var k = 0u;
    for (var j = 1u; j < cam.obj.y; j = j + 1u) {
      if (g >= insts[j].idx.x) { k = j; }
    }
    let e = insts[k];
    i = g - e.idx.x + e.idx.y;
    Ri = transpose(mat3x3<f32>(e.r0.xyz, e.r1.xyz, e.r2.xyz));
    ti = vec3<f32>(e.r0.w, e.r1.w, e.r2.w);
  }
  let pdi = pd[i];
  var pw = pdi.xyz;                                // centre in the world
  if (is_obj) { pw = Ri * pdi.xyz + ti; }
  let Rw2c = transpose(mat3x3<f32>(cam.r0.xyz, cam.r1.xyz, cam.r2.xyz));
  let tw2c = vec3<f32>(cam.r0.w, cam.r1.w, cam.r2.w);
  let pc = Rw2c * pw + tw2c;
  if (pc.z < 0.2) { return out; }
  let q = rot[i];
  let s = scl[i].xyz;
  let r = q.x; let x = q.y; let y = q.z; let z = q.w;
  var R = mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y)),
    vec3<f32>(2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x)),
    vec3<f32>(2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y)));
  if (is_obj) { R = Ri * R; }                      // Gaussian axes -> world
  // Projected 2D covariance (EWA), dilated by 0.3 px^2 as 3DGUT does; opacity compensated.
  let RS = mat3x3<f32>(R[0] * s.x, R[1] * s.y, R[2] * s.z);
  let M = Rw2c * RS;
  let V = M * transpose(M);
  let iz = 1.0 / pc.z;
  let K = cam.k;
  let j0 = vec3<f32>(K.x * iz, 0.0, -K.x * pc.x * iz * iz);
  let j1 = vec3<f32>(0.0, K.y * iz, -K.y * pc.y * iz * iz);
  let a = dot(j0, V * j0); let bb = dot(j0, V * j1); let c = dot(j1, V * j1);
  let ad = a + 0.3; let cd = c + 0.3;
  let detD = ad * cd - bb * bb;
  if (detD <= 0.0) { return out; }
  let det = a * c - bb * bb;
  let min_alpha = cam.res_minalpha.z;
  let opacity = pdi.w * sqrt(max(0.000025, det / detD));
  if (opacity < min_alpha) { return out; }
  // Screen extent: where the Gaussian's alpha falls below min_alpha (at most 3.33 sigma).
  let maxPow = log(opacity / min_alpha);
  let ef = min(3.33, sqrt(2.0 * maxPow));
  let mid = 0.5 * (ad + cd);
  let lambda = mid + sqrt(max(0.01, mid * mid - detD));
  let radius = ef * sqrt(lambda);
  let extent = min(ef * sqrt(vec2<f32>(ad, cd)), vec2<f32>(radius)) + 1.0;
  let center = vec2<f32>(K.x * pc.x * iz + K.z, K.y * pc.y * iz + K.w);
  let corner = vec2<f32>(f32(vid & 1u) * 2.0 - 1.0, f32((vid >> 1u) & 1u) * 2.0 - 1.0);
  let px = center + corner * extent;
  let res = cam.res_minalpha.xy;
  out.pos = vec4<f32>(px.x / res.x * 2.0 - 1.0, 1.0 - px.y / res.y * 2.0, 0.0, 1.0);  // row 0 at the top
  // The pixel's ray, in the Gaussian's frame scaled to unit sigma (interpolated over the quad).
  let dCam = vec3<f32>((px.x - K.z) / K.x, (px.y - K.w) / K.y, 1.0);
  let dWorld = transpose(Rw2c) * dCam;
  let Rt = transpose(R);
  out.dir = (Rt * dWorld) / s;
  out.oc = (Rt * (cam.cam_pos.xyz - pw)) / s;
  var dsh = normalize(pw - cam.cam_pos.xyz);
  if (is_obj) { dsh = transpose(Ri) * dsh; }       // view direction in the asset frame
  out.color = vec4<f32>(sh_color(i, dsh), pdi.w);
  return out;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
  // Response at the ray's closest approach to the Gaussian's centre (3DGUT's canonical ray).
  let d = normalize(v.dir);
  let g = cross(d, v.oc);
  let response = exp(-0.5 * dot(g, g));
  if (response <= cam.res_minalpha.w) { discard; }
  let alpha = min(cam.misc.x, response * v.color.a);
  if (alpha <= cam.res_minalpha.z) { discard; }
  return vec4<f32>(v.color.rgb * alpha, alpha);   // premultiplied
}
"""


# ----------------------------------------------------------------------------- splat file

def load_splat(path: str | Path, meta: dict) -> dict[str, np.ndarray]:
    """Read splat.bin (layout in docs/scene_format.md) into numpy arrays (views, no copies)."""
    n = int(meta['count'])
    raw = Path(path).read_bytes()
    offset = 0

    def take(dtype, count):
        nonlocal offset
        array = np.frombuffer(raw, dtype, count, offset)
        offset += array.nbytes
        return array

    arrays = {'position': take('<f4', 3 * n).reshape(n, 3), 'scale': take('<f4', 3 * n).reshape(n, 3),
              'rotation': take('<f4', 4 * n).reshape(n, 4), 'density': take('<f4', n),
              'sh': take('<f2', 48 * n).reshape(n, 48)}
    if offset != len(raw):
        raise ValueError(f'{path}: {len(raw)} bytes, expected {offset} for {n} particles')
    return arrays


# ----------------------------------------------------------------------------- device choice

_TYPE_RANK = {'DiscreteGPU': 0, 'IntegratedGPU': 1, 'VirtualGPU': 2, 'Unknown': 3, 'CPU': 4}


def _describe(adapter) -> str:
    info = adapter.info
    return f"{info.get('device') or info.get('vendor') or 'unknown'} ({info.get('adapter_type')}, {info.get('backend_type')})"


_BACKENDS_SET = False


def _native_backends_only() -> None:
    """Create the wgpu instance with the native backends only (Vulkan, Metal, DirectX 12), once.

    Probing wgpu's OpenGL backend initialises EGL/GLX, which aborts the process when Qt's OpenGL
    (the gym's 2D renderer) is already running in it; this renderer never uses OpenGL anyway.
    """
    global _BACKENDS_SET
    if _BACKENDS_SET:
        return
    _BACKENDS_SET = True
    try:
        from wgpu.backends.wgpu_native.extras import set_instance_extras
        set_instance_extras(backends=["Primary"])
    except Exception:   # another wgpu backend, or the instance exists already
        pass


def choose_adapter(device: Optional[str] = None):
    """Pick a wgpu adapter. device: 'auto' (default), 'gpu' or 'cpu' (see the module doc)."""
    import wgpu

    _native_backends_only()

    device = (device or os.environ.get('F1TENTH_SPLAT_DEVICE') or 'auto').lower()
    if device not in ('auto', 'gpu', 'cpu'):
        raise ValueError(f"render device must be 'auto', 'gpu' or 'cpu', not {device!r}")
    # Listing adapters makes the drivers log harmless notes (missing Vulkan extensions, EGL); keep
    # them out of the output (notebooks show them as errors) while still passing real errors through.
    import logging
    wgpu_log = logging.getLogger('wgpu')
    level = wgpu_log.level
    wgpu_log.setLevel(max(level, logging.ERROR))
    try:
        adapters = list(wgpu.gpu.enumerate_adapters_sync())
    finally:
        wgpu_log.setLevel(level)

    def rank(adapter):
        info = adapter.info
        # The OpenGL backend lacks storage buffers in vertex shaders on many drivers: last resort.
        return (_TYPE_RANK.get(info.get('adapter_type'), 3), info.get('backend_type') == 'OpenGL')

    adapters.sort(key=rank)
    gpus = [a for a in adapters if a.info.get('adapter_type') != 'CPU' and a.info.get('backend_type') != 'OpenGL']
    cpus = [a for a in adapters if a.info.get('adapter_type') == 'CPU']
    if device == 'cpu':
        candidates = cpus
    elif device == 'gpu':
        candidates = gpus
    else:
        candidates = gpus + cpus
    if not candidates:
        # Some platforms only expose the software adapter through request_adapter.
        if device != 'gpu':
            adapter = wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)
            if adapter is not None:
                return adapter
        found = ', '.join(_describe(a) for a in adapters) or 'none'
        hint = {'gpu': 'no GPU adapter found; try device="cpu"',
                'cpu': 'no software (CPU) adapter found; on Linux install Mesa lavapipe '
                       '(mesa-vulkan-drivers), on Windows WARP comes with DirectX 12',
                'auto': 'no usable adapter found; update your graphics drivers'}[device]
        raise RuntimeError(f'Renderer: {hint}. Adapters seen: {found}')
    return candidates[0]


# ----------------------------------------------------------------------------- renderer

class SplatRenderer:
    """Renders 960x600 RGB uint8 frames of a splat, optionally with objects, from camera-to-world poses.

    Example::

        renderer = SplatRenderer(scene.splat_path, scene.meta['splat'], intrinsics=scene.intrinsics)
        image = renderer.render(c2w)           # (600, 960, 3) uint8
        print(renderer.timing)                 # sort / GPU / total milliseconds of that frame

        car = renderer.add_asset(load_asset('path/to/car'))    # upload once -> asset id
        image = renderer.render(c2w, objects=[(car, pose_a), (car, pose_b)])   # 4x4 world_from_asset

    Objects (assets.py) are drawn with the scene's kernel and SH degree, depth-sorted together
    with the scene's particles. Without objects the frame is exactly the scene-only frame.
    """

    INSTANCE_FLOATS = 16   # per object instance in the instance buffer: 3 rows (R | t) + (start, base, 0, 0)

    def __init__(self, splat_path: str | Path, splat_meta: dict, width: int = 960, height: int = 600,
                 intrinsics: tuple[float, float, float, float] = DEFAULT_INTRINSICS, device: Optional[str] = None,
                 verbose: bool = True):
        import wgpu

        self.meta = splat_meta
        arrays = load_splat(splat_path, splat_meta)
        # Positions as three columns: the per-frame depth is then three multiply-adds (no BLAS call,
        # whose worker threads would spin on every core for this small product).
        self.columns = [np.ascontiguousarray(arrays['position'][:, i], np.float32) for i in range(3)]
        self.count = len(arrays['position'])
        self.width, self.height = width, height
        self.intrinsics = np.array(intrinsics, np.float32)

        self.adapter = choose_adapter(device)
        info = self.adapter.info
        self.device_name = _describe(self.adapter)
        self.is_cpu = info.get('adapter_type') == 'CPU'
        if verbose:
            note = '  (software rendering: expect 0.4-1.5 s per frame)' if self.is_cpu else ''
            print(f'Renderer: {self.device_name}{note}', flush=True)
        limits = self.adapter.limits
        self._max_binding = limits['max-storage-buffer-binding-size']
        needed = self.count * 48 * 2
        if needed > self._max_binding:
            raise RuntimeError(f'Renderer: the splat needs {needed / 1e6:.0f} MB storage buffers, '
                               f'{self.device_name} allows {self._max_binding / 1e6:.0f} MB')
        self.device = d = self.adapter.request_device_sync(required_limits={
            'max-storage-buffer-binding-size': limits['max-storage-buffer-binding-size'],
            'max-buffer-size': limits['max-buffer-size']})

        # Particle buffers: the scene's particles, then those of every added asset (asset frame).
        self._storage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        n = self.count
        self._particles = 0   # particles in the particle buffers (scene + assets)
        self._particle_buffers = self._upload_particles(arrays)
        self._camera_buffer = d.create_buffer(size=8 * 16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._order_buffer = d.create_buffer(size=4 * n, usage=self._storage)
        self._instance_buffer = d.create_buffer(size=8 * self.INSTANCE_FLOATS * 4, usage=self._storage)
        self._assets: list[dict] = []          # per asset id: path, base, count, columns
        self._asset_ids: dict[str, int] = {}   # resolved asset path -> id

        # Float16 accumulation target (as in the browser renderer).
        self._format = 'rgba16float'
        self._texture = d.create_texture(size=(width, height, 1), format=self._format,
                                         usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        module = d.create_shader_module(code=WGSL)
        blend = {'src_factor': 'one', 'dst_factor': 'one-minus-src-alpha', 'operation': 'add'}
        self._pipeline = d.create_render_pipeline(
            layout='auto',
            vertex={'module': module, 'entry_point': 'vs'},
            primitive={'topology': 'triangle-strip'},
            fragment={'module': module, 'entry_point': 'fs',
                      'targets': [{'format': self._format, 'blend': {'color': blend, 'alpha': blend}}]})
        self._make_bind_group()
        self.timing: dict[str, float] = {}

    # ------------------------------------------------------------------------- GPU buffers

    def _upload_particles(self, arrays: dict[str, np.ndarray], old: Optional[list] = None) -> list:
        """Append particles to the particle buffers (new, larger buffers; the old content is copied on the GPU)."""
        n = len(arrays['position'])
        pos_density = np.concatenate([arrays['position'], arrays['density'][:, None]], 1).astype(np.float32)
        scale = np.concatenate([arrays['scale'], np.zeros((n, 1), np.float32)], 1).astype(np.float32)
        rotation = np.ascontiguousarray(arrays['rotation'], np.float32)
        sh_words = np.ascontiguousarray(arrays['sh'], np.float16).view(np.uint32)   # two float16 per 32-bit word
        data = [pos_density, rotation, scale, sh_words]
        d = self.device
        if old is None:
            self._particles = n
            return [d.create_buffer_with_data(data=a.tobytes(), usage=self._storage) for a in data]
        total = self._particles + n
        if total * 96 > self._max_binding:
            raise RuntimeError(f'Renderer: scene + assets need {total * 96 / 1e6:.0f} MB storage buffers, '
                               f'{self.device_name} allows {self._max_binding / 1e6:.0f} MB')
        per_particle = [16, 16, 16, 96]
        new = [d.create_buffer(size=total * b, usage=self._storage) for b in per_particle]
        encoder = d.create_command_encoder()
        for src, dst in zip(old, new):
            encoder.copy_buffer_to_buffer(src, 0, dst, 0, src.size)
        d.queue.submit([encoder.finish()])
        for a, dst, b in zip(data, new, per_particle):
            d.queue.write_buffer(dst, self._particles * b, a.tobytes())
        for buffer in old:
            buffer.destroy()
        self._particles = total
        return new

    def _make_bind_group(self) -> None:
        buffers = [self._camera_buffer, *self._particle_buffers, self._order_buffer, self._instance_buffer]
        self._bind_group = self.device.create_bind_group(layout=self._pipeline.get_bind_group_layout(0), entries=[
            {'binding': i, 'resource': {'buffer': b, 'offset': 0, 'size': b.size}} for i, b in enumerate(buffers)])

    def _ensure(self, name: str, size: int) -> None:
        """Grow a per-frame buffer (order or instances) to at least `size` bytes (x1.5 headroom)."""
        buffer = getattr(self, name)
        if buffer.size >= size:
            return
        size = int(size * 1.5 + 15) // 16 * 16
        buffer.destroy()
        setattr(self, name, self.device.create_buffer(size=size, usage=self._storage))
        self._make_bind_group()

    # ------------------------------------------------------------------------- objects

    def add_asset(self, asset) -> int:
        """Upload an asset's Gaussians (asset frame) once; returns its id for render(objects=...).

        `asset`: an assets.Asset, or a name / path for assets.load_asset. Adding the same asset
        (same directory) again returns the same id without uploading.
        """
        if not hasattr(asset, 'splat_path'):
            from .scene import load_asset
            asset = load_asset(asset)
        key = str(Path(asset.path).resolve())
        if key in self._asset_ids:
            return self._asset_ids[key]
        arrays = load_splat(asset.splat_path, asset.splat_meta)
        base = self._particles
        self._particle_buffers = self._upload_particles(arrays, self._particle_buffers)
        self._make_bind_group()
        self._assets.append({'name': asset.name, 'path': key, 'base': base, 'count': len(arrays['position']),
                             'columns': [np.ascontiguousarray(arrays['position'][:, i], np.float32) for i in range(3)]})
        self._asset_ids[key] = len(self._assets) - 1
        return len(self._assets) - 1

    # ------------------------------------------------------------------------- frames

    def _scene_depth(self, c2w: np.ndarray) -> np.ndarray:
        rotation_w2c = c2w[:3, :3].T
        t_w2c = -rotation_w2c @ c2w[:3, 3]
        r = rotation_w2c[2].astype(np.float32)   # camera z axis (viewing direction) in the world
        x, y, z = self.columns
        return x * r[0] + y * r[1] + z * r[2] + np.float32(t_w2c[2])

    def sort(self, c2w: np.ndarray) -> np.ndarray:
        """Indices of the particles in front of the camera, far to near."""
        depth = self._scene_depth(c2w)
        visible = np.nonzero(depth >= MIN_DEPTH)[0]
        return visible[np.argsort(-depth[visible], kind='stable')].astype(np.uint32)

    def _object_table(self, objects) -> tuple[np.ndarray, list]:
        """Instance buffer contents for (asset_id, world_from_asset) pairs, and (start, asset, pose) per instance."""
        table = np.zeros((len(objects), self.INSTANCE_FLOATS), np.float32)
        ints = table.view(np.uint32)
        instances = []
        start = self.count
        for k, (asset_id, pose) in enumerate(objects):
            if not 0 <= int(asset_id) < len(self._assets):
                raise ValueError(f'unknown asset id {asset_id} (add_asset returned 0..{len(self._assets) - 1})')
            asset = self._assets[int(asset_id)]
            pose = np.asarray(pose, np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f'object {k}: pose must be a 4x4 matrix, not {pose.shape}')
            rotation = pose[:3, :3]
            if pose.shape != (4, 4) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) \
                    or np.linalg.det(rotation) < 0 or not np.isfinite(pose).all():
                raise ValueError(f'object {k}: pose must be a rigid 4x4 world_from_asset matrix')
            table[k, :12] = np.c_[rotation, pose[:3, 3]].ravel()
            ints[k, 12:14] = (start, asset['base'])
            instances.append((start, asset, pose))
            start += asset['count']
        return table, instances

    def sort_with_objects(self, c2w: np.ndarray, instances: list) -> np.ndarray:
        """Order values (see the WGSL) of the scene and object particles in front of the camera, far to near."""
        depth = self._scene_depth(c2w)
        visible = np.nonzero(depth >= MIN_DEPTH)[0]
        by_depth = np.argsort(-depth[visible], kind='stable')
        scene_order, scene_neg_depth = visible[by_depth].astype(np.uint32), -depth[visible][by_depth]
        rotation_w2c = c2w[:3, :3].T
        r = rotation_w2c[2]
        tz = -r @ c2w[:3, 3]
        ids, depths = [], []
        for start, asset, pose in instances:
            a = (r @ pose[:3, :3]).astype(np.float32)   # camera z axis in the asset frame
            x, y, z = asset['columns']
            dk = x * a[0] + y * a[1] + z * a[2] + np.float32(r @ pose[:3, 3] + tz)
            vis = np.nonzero(dk >= MIN_DEPTH)[0]
            ids.append((vis + start).astype(np.uint32))
            depths.append(dk[vis])
        ids, depths = np.concatenate(ids), np.concatenate(depths)
        by_depth = np.argsort(-depths, kind='stable')
        ids, neg = ids[by_depth], -depths[by_depth]
        # Merge two far-to-near lists (O(n)): object particles go after scene particles at equal depth.
        return np.insert(scene_order, np.searchsorted(scene_neg_depth, neg, side='right'), ids)

    def render(self, c2w: np.ndarray, objects: Optional[Sequence[tuple[int, np.ndarray]]] = None) -> np.ndarray:
        """Camera-to-world 4x4 (OpenCV axes: x right, y down, z forward) -> (H, W, 3) uint8 RGB.

        objects: optional (asset_id, world_from_asset 4x4) pairs (asset ids from add_asset); the same
        asset may appear several times. None or empty renders the scene alone (bit-identical).
        """
        c2w = np.asarray(c2w, np.float64)
        objects = list(objects) if objects is not None else []
        t0 = time.perf_counter()
        if objects:
            table, instances = self._object_table(objects)
            order = self.sort_with_objects(c2w, instances)
        else:
            order = self.sort(c2w)
        t1 = time.perf_counter()

        d = self.device
        if objects:
            self._ensure('_order_buffer', order.nbytes)
            self._ensure('_instance_buffer', table.nbytes)
            d.queue.write_buffer(self._instance_buffer, 0, table.tobytes())
        d.queue.write_buffer(self._order_buffer, 0, order.tobytes())
        rotation_w2c = c2w[:3, :3].T
        cam = np.zeros((8, 4), np.float32)
        cam[0:3, :3] = rotation_w2c
        cam[0:3, 3] = -rotation_w2c @ c2w[:3, 3]
        cam[3, :3] = c2w[:3, 3]
        cam[4] = self.intrinsics
        kernel = self.meta['kernel']
        cam[5] = (self.width, self.height, kernel['min_alpha'], kernel['min_response'])
        cam[6, :2] = (kernel['max_alpha'], self.meta['sh_degree'])
        cam.view(np.uint32)[7, :2] = (self.count, len(objects))
        d.queue.write_buffer(self._camera_buffer, 0, cam.tobytes())

        encoder = d.create_command_encoder()
        background = list(self.meta.get('background', [0.0, 0.0, 0.0])) + [0.0]
        render_pass = encoder.begin_render_pass(color_attachments=[{
            'view': self._texture.create_view(), 'load_op': 'clear', 'store_op': 'store',
            'clear_value': background}])
        render_pass.set_pipeline(self._pipeline)
        render_pass.set_bind_group(0, self._bind_group)
        render_pass.draw(4, len(order))
        render_pass.end()
        d.queue.submit([encoder.finish()])
        raw = d.queue.read_texture({'texture': self._texture, 'origin': (0, 0, 0), 'mip_level': 0},
                                   {'offset': 0, 'bytes_per_row': self.width * 8, 'rows_per_image': self.height},
                                   (self.width, self.height, 1))
        t2 = time.perf_counter()
        image = np.frombuffer(raw, np.float16).reshape(self.height, self.width, 4)[..., :3]
        out = np.round(np.clip(image.astype(np.float32), 0, 1) * 255).astype(np.uint8)
        self.timing = {'sort_ms': 1e3 * (t1 - t0), 'gpu_ms': 1e3 * (t2 - t1),
                       'total_ms': 1e3 * (time.perf_counter() - t0), 'visible': len(order), 'objects': len(objects)}
        return out
