"""Deterministic, offline two-plane stereo renderer; all distances in metres.

Run `python make.py` from the lab directory to write the compact dataset.
The notebook can render the same arrays in memory without writing anything.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def make_scene(seed=12):
    width, height = 384, 240
    K = np.float64([[400, 0, width / 2], [0, 400, height / 2], [0, 0, 1]])
    distortion = np.zeros(5)
    R = cv2.Rodrigues(np.float64([.012, .025, -.018]))[0]
    centre = np.float64([.12, 0, 0])
    T = -R @ centre
    rng = np.random.default_rng(seed)
    textures = [rng.integers(25, 231, (1024, 1024), dtype=np.uint8) for _ in range(2)]
    textures = [cv2.GaussianBlur(t, (3, 3), .6) for t in textures]
    v, u = np.indices((height, width), dtype=np.float64)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)

    def surface(camera_centre, rays):
        near = camera_centre + rays * ((2 - camera_centre[2]) / rays[..., 2])[..., None]
        foreground = ((near[..., 0] > -.18) & (near[..., 0] < .54)
                      & (near[..., 1] > -.36) & (near[..., 1] < .36))
        z = np.where(foreground, 2., 4.)
        xyz = camera_centre + rays * ((z - camera_centre[2]) / rays[..., 2])[..., None]
        return xyz, foreground

    def render(camera_centre, rays):
        xyz, foreground = surface(camera_centre, rays)
        tx = (xyz[..., 0] * 160 + 512).astype(np.float32)
        ty = (xyz[..., 1] * 160 + 512).astype(np.float32)
        layers = [cv2.remap(t, tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                  for t in textures]
        return np.where(foreground, layers[0], layers[1]).astype(np.uint8)

    rays = pixels @ np.linalg.inv(K).T
    left = render(np.zeros(3), rays)
    right = render(centre, rays @ R)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, distortion, K, distortion, (width, height), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    # Ground truth is ray-cast in the *rectified left* grid, independently of remap.
    rect_rays = pixels @ np.linalg.inv(P1[:, :3]).T @ R1
    world, foreground = surface(np.zeros(3), rect_rays)
    rect_xyz = world @ R1.T
    projected_right = rect_xyz @ P2[:, :3].T + P2[:, 3]
    right_uv = projected_right[..., :2] / projected_right[..., 2, None]
    gt = (u - right_uv[..., 0]).astype('<f4')
    # Visibility in the right view: the ray to a background point may hit the near panel.
    near_intersection = centre + (world - centre) * ((2 - centre[2]) / (world[..., 2] - centre[2]))[..., None]
    blocked = ((near_intersection[..., 0] > -.18) & (near_intersection[..., 0] < .54)
               & (near_intersection[..., 1] > -.36) & (near_intersection[..., 1] < .36)
               & ~foreground)
    visible = (~blocked & (right_uv[..., 0] >= 0) & (right_uv[..., 0] < width - 1)
               & (right_uv[..., 1] >= 0) & (right_uv[..., 1] < height - 1))
    # Fixed evaluation support for every configuration, never based on predictions.
    valid = visible & (u >= 80) & (u < width - 8) & (v >= 8) & (v < height - 8)
    points = world[24:-24:24, 96:-24:32].reshape(-1, 3)
    pts1 = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, distortion)[0].reshape(-1, 2)
    pts2 = cv2.projectPoints(points, cv2.Rodrigues(R)[0], T, K, distortion)[0].reshape(-1, 2)
    return dict(left=left, right=right, gt=gt, valid=valid, visible=visible,
                depth_gt=rect_xyz[..., 2].astype('<f4'), K=K, dist=distortion, R=R, T=T,
                size=(width, height), points=points, pts1=pts1, pts2=pts2)


def array_hashes(scene):
    """SHA-256 of C-order array bytes; float32 arrays use little-endian storage."""
    return {name: hashlib.sha256(np.ascontiguousarray(scene[name]).tobytes()).hexdigest()
            for name in ('left', 'right', 'gt', 'valid', 'depth_gt')}


if __name__ == '__main__':
    scene = make_scene()
    output = Path(__file__).resolve().parent / 'data'
    output.mkdir(exist_ok=True)
    cv2.imwrite(str(output / 'left-raw.png'), scene['left'])
    cv2.imwrite(str(output / 'right-raw.png'), scene['right'])
    np.savez_compressed(output / 'stereo.npz', **{k: v for k, v in scene.items() if k not in ('left', 'right')})
    print(json.dumps(array_hashes(scene), indent=2))
