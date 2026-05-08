import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (7, 5)


@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    color: np.ndarray


def normalize(v):
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        return arr / max(norm, 1e-12)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norm, 1e-12, None)


def intersect_sphere(ray_origin, ray_dir, sphere):
    oc = ray_origin - sphere.center
    a = np.sum(ray_dir * ray_dir, axis=-1)
    b = 2.0 * np.sum(oc * ray_dir, axis=-1)
    c = np.sum(oc * oc, axis=-1) - sphere.radius ** 2
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


def render_scene(width=320, height=180):
    aspect = width / height
    camera_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    x = np.linspace(-aspect, aspect, width)
    y = np.linspace(1.0, -1.0, height)
    xx, yy = np.meshgrid(x, y)
    pixel_positions = np.stack([xx, yy, -1.5 * np.ones_like(xx)], axis=-1)
    ray_dirs = normalize(pixel_positions - camera_origin)

    spheres = [
        Sphere(np.array([0.0, 0.0, -3.5]), 0.8, np.array([0.85, 0.35, 0.25])),
        Sphere(np.array([1.2, -0.15, -4.3]), 0.9, np.array([0.25, 0.65, 0.95])),
        Sphere(np.array([-1.4, -0.35, -4.0]), 0.7, np.array([0.7, 0.8, 0.3])),
    ]

    light_dir = normalize(np.array([0.8, 1.2, -0.6], dtype=np.float64))
    image = np.zeros((height, width, 3), dtype=np.float64)
    closest_t = np.full((height, width), np.inf, dtype=np.float64)
    intersection_tests = 0

    for sphere in spheres:
        t = intersect_sphere(camera_origin, ray_dirs, sphere)
        intersection_tests += width * height
        hit = t < closest_t
        if not np.any(hit):
            continue

        hit_points = camera_origin + ray_dirs[hit] * t[hit, None]
        normals = normalize(hit_points - sphere.center)
        lambert = np.clip(np.sum(normals * light_dir, axis=-1), 0.0, 1.0)
        shade = 0.15 + 0.85 * lambert
        image[hit] = sphere.color * shade[:, None]
        closest_t[hit] = t[hit]

    sky = np.stack([
        0.15 + 0.35 * (yy + 1.0) / 2.0,
        0.2 + 0.45 * (yy + 1.0) / 2.0,
        0.3 + 0.55 * (yy + 1.0) / 2.0,
    ], axis=-1)
    miss = ~np.isfinite(closest_t)
    image[miss] = sky[miss]

    return np.clip(image, 0.0, 1.0), intersection_tests


def timing_experiment(resolutions=(64, 128, 192, 256, 320)):
    rows = []
    for width in resolutions:
        height = max(1, width * 9 // 16)
        start = time.perf_counter()
        _, tests = render_scene(width=width, height=height)
        elapsed = time.perf_counter() - start
        rows.append((width, height, tests, elapsed))
    return rows


def main():
    start = time.perf_counter()
    image, intersection_tests = render_scene()
    elapsed = time.perf_counter() - start

    plt.imshow(image)
    plt.title(f'Python baseline ray tracer\n{intersection_tests:,} sphere tests in {elapsed:.3f} s')
    plt.axis('off')
    plt.show()

    print(f'Intersection tests: {intersection_tests:,}')
    print(f'Elapsed time:       {elapsed:.3f} s')

    results = timing_experiment()
    for width, height, tests, sample_elapsed in results:
        print(f'{width:>4} x {height:<4}  tests={tests:>9,}  time={sample_elapsed:>7.4f} s')

    plt.plot([row[0] for row in results], [row[3] for row in results], marker='o')
    plt.xlabel('Image width')
    plt.ylabel('Render time [s]')
    plt.title('Baseline scaling without BVH or RT-core acceleration')
    plt.grid(True, alpha=0.3)
    plt.show()

    return {
        'image': image,
        'intersection_tests': intersection_tests,
        'elapsed': elapsed,
        'results': results,
    }


if __name__ == '__main__':
    globals().update(main())
