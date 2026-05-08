import ctypes
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


def normalize(v):
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        return arr / max(norm, 1e-12)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norm, 1e-12, None)


def intersect_sphere(ray_origin, ray_dir, center, radius):
    oc = ray_origin - center
    a = np.sum(ray_dir * ray_dir, axis=-1)
    b = 2.0 * np.sum(oc * ray_dir, axis=-1)
    c = np.sum(oc * oc, axis=-1) - radius ** 2
    discriminant = b * b - 4.0 * a * c

    hit_mask = discriminant >= 0.0
    t = np.full(ray_dir.shape[:-1], np.inf, dtype=np.float64)

    if np.any(hit_mask):
        sqrt_disc = np.sqrt(discriminant[hit_mask])
        a_hit = a[hit_mask]
        t0 = (-b[hit_mask] - sqrt_disc) / (2.0 * a_hit)
        t1 = (-b[hit_mask] + sqrt_disc) / (2.0 * a_hit)
        t_hit = np.where(t0 > 1e-4, t0, t1)
        t[hit_mask] = np.where(t_hit > 1e-4, t_hit, np.inf)

    return t


def render_scene_cpu(width=160, height=90):
    aspect = width / height
    camera_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    x = np.linspace(-aspect, aspect, width)
    y = np.linspace(1.0, -1.0, height)
    xx, yy = np.meshgrid(x, y)
    pixel_positions = np.stack([xx, yy, -1.5 * np.ones_like(xx)], axis=-1)
    ray_dirs = normalize(pixel_positions - camera_origin)

    sphere_centers = [
        np.array([0.0, 0.0, -3.5], dtype=np.float64),
        np.array([1.2, -0.15, -4.3], dtype=np.float64),
        np.array([-1.4, -0.35, -4.0], dtype=np.float64),
    ]
    sphere_radii = [0.8, 0.9, 0.7]
    sphere_colors = [
        np.array([0.85, 0.35, 0.25], dtype=np.float64),
        np.array([0.25, 0.65, 0.95], dtype=np.float64),
        np.array([0.70, 0.80, 0.30], dtype=np.float64),
    ]

    light_dir = normalize(np.array([0.8, 1.2, -0.6], dtype=np.float64))
    image = np.zeros((height, width, 3), dtype=np.float64)
    closest_t = np.full((height, width), np.inf, dtype=np.float64)

    for center, radius, color in zip(sphere_centers, sphere_radii, sphere_colors):
        t = intersect_sphere(camera_origin, ray_dirs, center, radius)
        hit = t < closest_t
        if not np.any(hit):
            continue

        hit_points = camera_origin + ray_dirs[hit] * t[hit, None]
        normals = normalize(hit_points - center)
        lambert = np.clip(np.sum(normals * light_dir, axis=-1), 0.0, 1.0)
        shade = 0.15 + 0.85 * lambert
        image[hit] = color * shade[:, None]
        closest_t[hit] = t[hit]

    sky = np.stack([
        0.15 + 0.35 * (yy + 1.0) / 2.0,
        0.20 + 0.45 * (yy + 1.0) / 2.0,
        0.30 + 0.55 * (yy + 1.0) / 2.0,
    ], axis=-1)
    miss = ~np.isfinite(closest_t)
    image[miss] = sky[miss]
    return np.clip(image, 0.0, 1.0)


def has_nvidia_driver():
    if os.name == "nt":
        try:
            ctypes.WinDLL("nvcuda.dll")
            return True
        except OSError:
            return False

    for candidate in ("libcuda.so", "libcuda.so.1"):
        try:
            ctypes.CDLL(candidate)
            return True
        except OSError:
            continue
    return False


if "numba.cuda" not in sys.modules and not has_nvidia_driver():
    os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

from numba import cuda


def select_cuda_backend():
    if cuda.is_available() and os.environ.get("NUMBA_ENABLE_CUDASIM") != "1":
        return "NVIDIA GPU"
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        return "Numba CUDA simulator"
    raise RuntimeError(
        "CUDA is not available and the simulator was not enabled before numba.cuda was imported. "
        "Restart the kernel and run the notebook cell again."
    )


@cuda.jit
def render_kernel(image, sphere_centers, sphere_radii, sphere_colors, light_dir, width, height):
    pixel_x, pixel_y = cuda.grid(2)
    if pixel_x >= width or pixel_y >= height:
        return

    aspect = width / height
    screen_x = -aspect + (2.0 * aspect) * pixel_x / max(width - 1, 1)
    screen_y = 1.0 - 2.0 * pixel_y / max(height - 1, 1)
    dir_x = screen_x
    dir_y = screen_y
    dir_z = -1.5

    inv_len = 1.0 / math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    dir_x *= inv_len
    dir_y *= inv_len
    dir_z *= inv_len

    closest_t = 1e20
    out_r = 0.0
    out_g = 0.0
    out_b = 0.0

    for sphere_idx in range(sphere_radii.shape[0]):
        center_x = sphere_centers[sphere_idx, 0]
        center_y = sphere_centers[sphere_idx, 1]
        center_z = sphere_centers[sphere_idx, 2]
        radius = sphere_radii[sphere_idx]

        oc_x = -center_x
        oc_y = -center_y
        oc_z = -center_z
        b = 2.0 * (oc_x * dir_x + oc_y * dir_y + oc_z * dir_z)
        c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - radius * radius
        discriminant = b * b - 4.0 * c
        if discriminant < 0.0:
            continue

        sqrt_disc = math.sqrt(discriminant)
        t0 = (-b - sqrt_disc) * 0.5
        t1 = (-b + sqrt_disc) * 0.5
        t = t0 if t0 > 1e-4 else t1
        if t <= 1e-4 or t >= closest_t:
            continue

        hit_x = dir_x * t
        hit_y = dir_y * t
        hit_z = dir_z * t
        normal_x = (hit_x - center_x) / radius
        normal_y = (hit_y - center_y) / radius
        normal_z = (hit_z - center_z) / radius

        lambert = normal_x * light_dir[0] + normal_y * light_dir[1] + normal_z * light_dir[2]
        if lambert < 0.0:
            lambert = 0.0
        shade = 0.15 + 0.85 * lambert

        out_r = sphere_colors[sphere_idx, 0] * shade
        out_g = sphere_colors[sphere_idx, 1] * shade
        out_b = sphere_colors[sphere_idx, 2] * shade
        closest_t = t

    if closest_t >= 1e19:
        out_r = 0.15 + 0.35 * (screen_y + 1.0) * 0.5
        out_g = 0.20 + 0.45 * (screen_y + 1.0) * 0.5
        out_b = 0.30 + 0.55 * (screen_y + 1.0) * 0.5

    image[pixel_y, pixel_x, 0] = out_r
    image[pixel_y, pixel_x, 1] = out_g
    image[pixel_y, pixel_x, 2] = out_b


def render_scene_cuda(width=160, height=90):
    sphere_centers = np.array([
        [0.0, 0.0, -3.5],
        [1.2, -0.15, -4.3],
        [-1.4, -0.35, -4.0],
    ], dtype=np.float32)
    sphere_radii = np.array([0.8, 0.9, 0.7], dtype=np.float32)
    sphere_colors = np.array([
        [0.85, 0.35, 0.25],
        [0.25, 0.65, 0.95],
        [0.70, 0.80, 0.30],
    ], dtype=np.float32)
    light_dir = np.array([0.5121475, 0.76822126, -0.38411063], dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (width + threads_per_block[0] - 1) // threads_per_block[0],
        (height + threads_per_block[1] - 1) // threads_per_block[1],
    )

    start = time.perf_counter()
    d_image = cuda.to_device(image)
    d_centers = cuda.to_device(sphere_centers)
    d_radii = cuda.to_device(sphere_radii)
    d_colors = cuda.to_device(sphere_colors)
    d_light = cuda.to_device(light_dir)

    render_kernel[blocks_per_grid, threads_per_block](
        d_image, d_centers, d_radii, d_colors, d_light, width, height
    )
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    return d_image.copy_to_host(), elapsed, blocks_per_grid, threads_per_block


def main():
    backend = select_cuda_backend()
    gpu_image, gpu_elapsed, blocks_per_grid, threads_per_block = render_scene_cuda()
    reference_image = render_scene_cpu(width=gpu_image.shape[1], height=gpu_image.shape[0])

    mean_abs_diff = np.mean(np.abs(reference_image - gpu_image))
    max_abs_diff = np.max(np.abs(reference_image - gpu_image))
    intersection_tests = gpu_image.shape[0] * gpu_image.shape[1] * 3

    plt.imshow(np.clip(gpu_image, 0.0, 1.0))
    plt.title(
        f"CUDA ray tracer on {backend}\n"
        f"launch={blocks_per_grid} blocks x {threads_per_block} threads, time={gpu_elapsed:.3f} s"
    )
    plt.axis("off")
    plt.show()

    print(f"Backend:              {backend}")
    print(f"Image size:           {gpu_image.shape[1]} x {gpu_image.shape[0]}")
    print(f"Intersection tests:   {intersection_tests:,}")
    print(f"Elapsed time:         {gpu_elapsed:.3f} s")
    print(f"Mean abs diff vs CPU: {mean_abs_diff:.3e}")
    print(f"Max abs diff vs CPU:  {max_abs_diff:.3e}")

    return {
        "backend": backend,
        "gpu_image": gpu_image,
        "gpu_elapsed": gpu_elapsed,
        "blocks_per_grid": blocks_per_grid,
        "threads_per_block": threads_per_block,
        "reference_image": reference_image,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
    }


if __name__ == "__main__":
    globals().update(main())
