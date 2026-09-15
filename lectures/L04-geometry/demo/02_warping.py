# -*- coding: utf-8 -*-
"""L4, демо-блок 2: ресемплинг — прямой и обратный варп, интерполяция, алиасинг.

Запуск из корня репозитория:

    python lectures/L04-geometry/demo/02_warping.py

Каждый фрагмент — ровно тот код, что стоит на слайде; вывод — те числа, что
стоят в @result. Данные рядом со скриптом:

    iu-building.jpg   фотография здания ИУ, 1920 × 1279 (уменьшается до 960 × 640)
    text-page.png     страница текста 640 × 480 — для сравнения интерполяций

Тайминги (демо 2.2, 2.3) зависят от машины — на слайде порядок величины;
PSNR в демо 2.4 считается против эталона Lanczos из Pillow, как в демо 2.2 лекции 1.
"""
import os
import sys
import timeit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # корень репозитория

import cv2                                  # noqa: E402
import numpy as np                          # noqa: E402
from cvcourse import io as cio              # noqa: E402
from cvcourse.metrics import psnr           # noqa: E402


def data(name):
    return os.path.join(HERE, name)


ms = lambda fn: 1e3 * min(timeit.repeat(fn, number=1, repeat=5))   # noqa: E731


def demo_2_1(img):
    print("\n--- Демо 2.1 — прямой варп: дыры")
    src = cv2.resize(img, (960, 640), interpolation=cv2.INTER_AREA)
    h, w = src.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 20, 1.6)
    c = cv2.transform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), M)[0]
    M[:, 2] -= np.floor(c.min(0))                     # холст по повёрнутым углам
    size = tuple(np.int32(np.ceil(c.max(0) - np.floor(c.min(0)))))
    p = np.c_[np.mgrid[:h, :w][::-1].reshape(2, -1).T, np.ones(h * w)] @ M.T
    xd, yd = np.round(p).astype(int).T                # forward: куда летит пиксель
    hit = np.zeros((size[1] + 1, size[0] + 1), bool); hit[yd, xd] = True
    region = cv2.warpAffine(np.ones((h, w), np.uint8), M, size) > 0
    holes = 100 * (~hit[:size[1], :size[0]] & region).sum() / region.sum()
    print("прямой варп: %.0f %% дыр; обратный (warpAffine) — 0 %%" % holes)
    return src


def demo_2_2(src):
    print("\n--- Демо 2.2 — свой билинейный remap")

    def bilinear(img, mx, my):                       # ⭐ звёздочка занятия 4
        x0, y0 = np.floor(mx).astype(int), np.floor(my).astype(int)
        fx, fy = (mx - x0)[..., None], (my - y0)[..., None]
        top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
        bot = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
        return np.uint8(np.round(top * (1 - fy) + bot * fy))
    A = cv2.invertAffineTransform(cv2.getRotationMatrix2D((480, 320), 10, 1.3))
    ys, xs = np.mgrid[:640, :960].astype(np.float32)
    mx, my = [np.float32(a * xs + b * ys + c) for a, b, c in A]  # dst → src
    ref = cv2.remap(src, mx, my, cv2.INTER_LINEAR)
    d = np.abs(bilinear(src, mx, my).astype(int) - ref)
    print("против cv2.remap: макс %d, среднее %.2f кода" % (d.max(), d.mean()))
    # строка сверх слайда: время того же вызова — на слайде порядок величины
    print("время: %.0f мс против %.1f мс у cv2.remap"
          % (ms(lambda: bilinear(src, mx, my)),
             ms(lambda: cv2.remap(src, mx, my, cv2.INTER_LINEAR))))


def demo_2_3(src):
    print("\n--- Демо 2.3 — четыре интерполяции в числах")
    page = cio.imread(data("text-page.png"), "gray")                       # 640 × 480
    small = cv2.resize(page, (160, 120), interpolation=cv2.INTER_AREA)     # текст ×4 вниз
    crop = src[230:350, 330:490]                                           # фасад 160 × 120
    tiny = cv2.resize(crop, (20, 15), interpolation=cv2.INTER_AREA)        # фото ×8 вниз
    for name in ("INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LANCZOS4"):
        f = getattr(cv2, name)
        up_t = cv2.resize(small, (640, 480), interpolation=f)
        up_p = cv2.resize(tiny, (160, 120), interpolation=f)
        print("%-15s текст ×4: %.1f дБ · фото ×8: %.1f дБ · %.2f мс"
              % (name, psnr(page, up_t), psnr(crop, up_p),
                 ms(lambda: cv2.resize(small, (640, 480), interpolation=f))))


def demo_2_4(img):
    print("\n--- Демо 2.4 — уменьшение: «AREA» в варпе не работает")
    from PIL import Image
    h = img.shape[0]
    size = (480, 320)                                  # ×¼ от 1920 × 1279
    pil = Image.fromarray(img[..., ::-1]).resize(size, Image.LANCZOS)
    ref = np.asarray(pil)[..., ::-1]                   # чужой эталон, как в L1
    M = np.float32([[0.25, 0, -0.375], [0, 320 / h, (320 / h - 1) / 2]])
    lin = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR)
    area = cv2.warpAffine(img, M, size, flags=cv2.INTER_AREA)    # молча LINEAR
    blur = cv2.warpAffine(cv2.GaussianBlur(img, (0, 0), 1.5), M, size)
    d = np.abs(lin.astype(int) - area).max()
    print("warpAffine LINEAR %.1f дБ · «AREA» %.1f дБ · Δ между ними %d кодов"
          % (psnr(ref, lin), psnr(ref, area), d))
    print("GaussianBlur σ 1.5 → warpAffine LINEAR: %.1f дБ" % psnr(ref, blur))


if __name__ == "__main__":
    img = cio.imread(data("iu-building.jpg"))
    src = demo_2_1(img)
    demo_2_2(src)
    demo_2_3(src)
    demo_2_4(img)
