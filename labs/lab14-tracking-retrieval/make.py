"""Offline procedural images and ready SIFT descriptors for lab14.

No fitted vocabulary, search results or association solutions live here.
Train images, database and queries are separate. GT is for evaluation only.
The NPZ archive uses fixed ZIP timestamps and sorted members.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np


def poster(seed):
    """L14 demo poster construction, reused with attribution in data/README."""
    rng = np.random.default_rng(seed)
    image = np.full((224, 224), 235, np.uint8)
    for gy in range(4):
        for gx in range(4):
            x, y = gx * 56, gy * 56
            patch = rng.integers(80, 240, (40, 40), dtype=np.uint8)
            image[y+8:y+48, x+8:x+48] = cv2.GaussianBlur(patch, (3, 3), .5)
            for _ in range(4):
                a, b = rng.integers(8, 48, (2, 2))
                cv2.line(image, (x+int(a[0]), y+int(a[1])),
                         (x+int(b[0]), y+int(b[1])), int(rng.integers(10, 90)), 2)
            cv2.putText(image, chr(65+int(rng.integers(0, 26))), (x+12, y+40),
                        0, .8, 20, 2, cv2.LINE_AA)
    return image


def features(image):
    points, desc = cv2.SIFT_create(nfeatures=350, contrastThreshold=.025).detectAndCompute(image, None)
    xy = np.float32([p.pt for p in points]).reshape(-1, 2)
    return xy, desc if desc is not None else np.empty((0, 128), np.float32)


def make_scene(seed=14):
    """Return arrays only; canonical default shares the L14 retrieval scene."""
    rng = np.random.default_rng(seed)
    database = [poster(20 + i + (seed-14)*10) for i in range(6)]
    original, scrambled = database[0], database[0].copy()
    order = rng.permutation(16)
    for j, source in enumerate(order):
        yy, xx = divmod(j, 4)
        sy, sx = divmod(int(source), 4)
        scrambled[yy*56:(yy+1)*56, xx*56:(xx+1)*56] = original[sy*56:(sy+1)*56, sx*56:(sx+1)*56]
    database.append(scrambled)
    H = np.array([[.94, .05, 10], [-.03, 1.02, 16], [.00012, -.00015, 1.]])
    result = dict(database=np.stack(database), query=cv2.warpPerspective(original, H, (250, 250), borderValue=245),
                  foreign=poster(900+seed), blank=np.full((250, 250), 235, np.uint8),
                  query_H=H, permutation=order.astype('<i4'))
    train = [poster(100+i+(seed-14)*10) for i in range(6)]
    result['train_images'] = np.stack(train)
    result['train_desc'] = np.vstack([features(im)[1] for im in train])
    for name, image in [(f'db{i}', im) for i, im in enumerate(database)] + [
            (name, result[name]) for name in ('query', 'foreign', 'blank')]:
        result[name+'_xy'], result[name+'_desc'] = features(image)
    color_frames, color_gt, frames, boxes = [], [], [], []
    visible = np.ones((40, 2), bool)
    visible[7:9, 0] = False
    for t in range(40):
        color = np.full((190, 300, 3), (70, 100, 60), np.uint8)
        cx, cy = 55+3*t, 95+int(8*np.sin(t/10))
        rx, ry = 20+int(.45*t), 15+int(.3*t)
        cv2.ellipse(color, (cx, cy), (rx, ry), 0, 0, 360, (20, 65, 230), -1)
        color_frames.append(color)
        color_gt.append((cx-rx, cy-ry, 2*rx+1, 2*ry+1))
        frame = np.full((140, 280, 3), 50, np.uint8)
        pair = [(20+5*t, 45, 30, 25), (220-5*t, 53, 30, 25)]
        for j, (x, y, w, h) in enumerate(pair):
            if visible[t, j]:
                cv2.rectangle(frame, (x, y), (x+w-1, y+h-1), (30, 50, 230), -1)
        frames.append(frame)
        boxes.append(pair)
    result.update(color_frames=np.stack(color_frames), color_gt=np.array(color_gt, dtype='<f8'),
                  frames=np.stack(frames), boxes_gt=np.array(boxes, dtype='<f8'), visible=visible)
    return result


def array_hashes(scene):
    return {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in sorted(scene.items())}


def export(output, scene):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output/'scene.npz', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(scene.items()):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, value, allow_pickle=False)
            info = zipfile.ZipInfo(name+'.npy', date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buf.getvalue(), compresslevel=6)
    hashes = array_hashes(scene)
    manifest = dict(seed=14, arrays={name: dict(shape=list(value.shape), dtype=value.dtype.str, sha256=hashes[name])
                                   for name, value in sorted(scene.items())},
                    archive_sha256=hashlib.sha256((output/'scene.npz').read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'data')
    args = parser.parse_args()
    assert cv2.__version__ == '4.14.0', 'Use iu-intro-cv/.venv'
    cv2.setNumThreads(1)
    print(json.dumps(export(args.output, make_scene()), indent=2))
