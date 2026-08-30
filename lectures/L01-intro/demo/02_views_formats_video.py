# -*- coding: utf-8 -*-
"""L1, демо-блок 2: срезы, ресемплинг, сжатие, видео — четыре фрагмента с лекции.

Запуск из корня репозитория:

    python lectures/L01-intro/demo/02_views_formats_video.py

Синтетический ролик 1080p пишется во временный каталог при первом запуске —
никаких внешних данных не нужно. Скорость декодирования (демо 2.4) зависит
от машины: на слайде порядок величины.
"""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import cv2                                   # noqa: E402
import numpy as np                           # noqa: E402
from PIL import Image                        # noqa: E402
from cvcourse import io as cio, metrics      # noqa: E402

PHOTO = os.path.join(HERE, "iu-building.jpg")


def demo_2_1(img):
    print("\n--- Демо 2.1 — срез это вид на память")
    roi = img[300:500, 600:900]        # y: 300…499, x: 600…899
    print(roi.shape, roi.base is img)  # вид, а не копия

    before = img[300, 600].copy()
    roi[:] = 255 - roi                 # инвертировали ТОЛЬКО срез…
    print("…но img[300, 600] был", before, "стал", img[300, 600])

    patch = img[300:500, 600:900].copy()   # копия живёт отдельно
    patch[:] = 0
    print("после patch[:] = 0 в img:", img[300, 600])
    roi[:] = 255 - roi                 # вернуть кадр как был


def demo_2_2(img):
    print("\n--- Демо 2.2 — флаг resize в числах (эталон: Lanczos из Pillow)")
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for k in (1.1, 4.0):
        size = (round(img.shape[1] / k), round(img.shape[0] / k))
        ref = np.asarray(pil.resize(size, Image.LANCZOS))  # чужой эталон
        ref = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
        for name in ("INTER_LINEAR", "INTER_AREA", "INTER_CUBIC"):
            f = getattr(cv2, name)
            got = cv2.resize(img, size, interpolation=f)
            print(k, name, "%.1f дБ" % metrics.psnr(ref, got))


def demo_2_3(img):
    print("\n--- Демо 2.3 — сжатие в числах")
    print("сырой массив: %d КБ" % (img.nbytes // 1024))
    ok, png = cv2.imencode(".png", img)          # в память, не на диск
    print("PNG: %d КБ, без потерь" % (len(png) // 1024))
    for q in (95, 75, 50, 20):
        prm = [cv2.IMWRITE_JPEG_QUALITY, q]
        ok, buf = cv2.imencode(".jpg", img, prm)
        back = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        changed = (back != img).any(axis=2).mean() * 100
        kb, ps = len(buf) // 1024, metrics.psnr(img, back)
        print("q=%d: %d КБ, PSNR %.1f дБ, изменено %.0f %%"
              % (q, kb, ps, changed))


def synth_video(path, frames=90, size=(1920, 1080), fps=30):
    """Ролик: движущийся прямоугольник и шум — чтобы кодеку было что сжимать."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    for t in range(frames):
        fr = np.full((size[1], size[0], 3), 200, np.uint8)
        cv2.rectangle(fr, (100 + 12 * t, 300), (500 + 12 * t, 700), (60, 70, 200), -1)
        fr = np.clip(fr.astype(np.int16) + rng.normal(0, 4, fr.shape), 0, 255).astype(np.uint8)
        vw.write(fr)
    vw.release()
    return path


def demo_2_4():
    print("\n--- Демо 2.4 — видео: заголовок против измерения")
    path = os.path.join(tempfile.gettempdir(), "l01-synth-1080p.mp4")
    if not os.path.exists(path):
        synth_video(path)                       # 90 кадров, 1080p
    print(cio.video_info(path))                 # что написано в заголовке
    t0, n = time.perf_counter(), 0
    for frame in cio.video_frames(path):        # захват освободится сам
        n += 1                                  # здесь была бы обработка
    dt = time.perf_counter() - t0
    raw_mb = n * frame.size / 2**20             # H · W · 3 байт на кадр
    print("%d кадров за %.2f с → %.0f к/с" % (n, dt, n / dt))
    mb = os.path.getsize(path) / 2**20
    print("сырых %.0f МБ, файл %.1f МБ → сжатие в %.0f раз" % (raw_mb, mb, raw_mb / mb))


if __name__ == "__main__":
    image = cio.imread(PHOTO)
    demo_2_1(image)
    demo_2_2(image)
    demo_2_3(image)
    demo_2_4()
