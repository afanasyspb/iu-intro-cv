"""Deterministic offline scenes for motion masks and known image translations.

No video codec, camera, network or lecture assets are required. Canonical hashes
refer to C-order array bytes (not ZIP timestamps). Run to export scenes.npz and
manifest.json; notebooks use make_scene() directly and need no generated files.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def translate(image, shift):
    """Forward translation: target(x+u,y+v)=source(x,y); no cyclic wrapping."""
    matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def make_scene(seed=13, shift=(6., 3.)):
    rng = np.random.default_rng(seed)
    background = rng.integers(80, 170, (96, 144, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(background, (5, 5), 1)
    video, masks, shadows = [], [], []
    for t in range(80):
        frame = np.clip(background.astype(np.int16) + rng.integers(-2, 3, background.shape), 0, 255).astype(np.uint8)
        mask = np.zeros((96, 144), bool)
        shadow = np.zeros_like(mask)
        if t >= 24:
            x, y = 12 + 2 * min(t - 24, 22), 32
            shadow[y + 9:y + 31, x + 9:x + 29] = True
            frame[shadow] = np.uint8(background[shadow] * .65)
            mask[y:y + 22, x:x + 20] = True
            frame[mask] = (35, 55, 225)
            shadow &= ~mask
        # Isolated sensor impulses: morphology should remove most of these.
        rows, cols = rng.integers(0, 96, 18), rng.integers(0, 144, 18)
        frame[rows, cols] = 255
        video.append(frame)
        masks.append(mask)
        shadows.append(shadow)
    a = rng.integers(20, 236, (128, 176), dtype=np.uint8)
    a = cv2.GaussianBlur(a, (5, 5), .9)
    coarse = rng.integers(20, 236, (8, 11), dtype=np.uint8)
    coarse = cv2.resize(coarse, (176, 128), interpolation=cv2.INTER_CUBIC)
    a = cv2.addWeighted(a, .45, coarse, .55, 0)
    cv2.rectangle(a, (41, 34), (68, 59), 205, 2)
    cv2.circle(a, (119, 83), 13, 45, 2)
    shift = np.array(shift, dtype='<f4')
    b = translate(a, shift)
    gt = np.broadcast_to(shift, (*a.shape, 2)).copy()
    yy, xx = np.indices(a.shape)
    # Fixed source-domain support, including a full patch margin in both images.
    valid = ((xx >= 22) & (xx < a.shape[1] - 22) & (yy >= 22) & (yy < a.shape[0] - 22)
             & (xx + shift[0] >= 22) & (xx + shift[0] < a.shape[1] - 22)
             & (yy + shift[1] >= 22) & (yy + shift[1] < a.shape[0] - 22))
    occluded = b.copy()
    occluded[40:88, 70:114] = 125
    visible = valid & ~((xx + shift[0] >= 70) & (xx + shift[0] < 114)
                        & (yy + shift[1] >= 40) & (yy + shift[1] < 88))
    # Infinite vertical edge: shifting parallel to it leaves the image identical.
    edge = np.full((64, 80), 40, np.uint8)
    edge[:, 40:] = 210
    return dict(video=np.stack(video), video_gt=np.stack(masks), shadow_gt=np.stack(shadows),
                a=a, b=b, gt=gt, valid=valid, occluded=occluded, visible=visible,
                shift=shift, edge=edge)


def array_hashes(scene):
    return {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in sorted(scene.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data')
    args = parser.parse_args()
    assert cv2.__version__ == '4.14.0', 'Use the course environment'
    cv2.setNumThreads(1)
    scene = make_scene()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'scenes.npz', **scene)
    manifest = {name: dict(shape=list(value.shape), dtype=value.dtype.str, sha256=array_hashes(scene)[name])
                for name, value in sorted(scene.items())}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
