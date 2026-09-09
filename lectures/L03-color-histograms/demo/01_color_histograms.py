# -*- coding: utf-8 -*-
"""L3, демо-блок 1: цвет, точечные операции и гистограммы — четыре фрагмента с лекции.

Запуск из корня репозитория:

    python lectures/L03-color-histograms/demo/01_color_histograms.py

Каждый фрагмент — ровно тот код, что стоит на слайде; вывод — те числа, что
стоят в @result. Фото — здание Университета Иннополис (1920 × 1279).
Тайминги демо 1.2 зависят от машины — на слайде порядок величины.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # корень репозитория

import cv2                                  # noqa: E402
import numpy as np                          # noqa: E402
from cvcourse import io as cio              # noqa: E402

PHOTO = os.path.join(HERE, "iu-building.jpg")


def demo_1_1():
    print("\n--- Демо 1.1 — три пространства, одно фото")
    img = cio.imread(PHOTO)
    b, g, r = [c.ravel().astype(np.float32) for c in cv2.split(img)]
    cc = lambda p, q: np.corrcoef(p, q)[0, 1]
    print("corr(R,G) = %.2f, corr(G,B) = %.2f, corr(R,B) = %.2f"
          % (cc(r, g), cc(g, b), cc(r, b)))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, v = [hsv[..., i].ravel().astype(np.float32) for i in (0, 2)]
    print("corr(H,V) = %.2f — оттенок не знает о яркости" % cc(h, v))
    print("hue max = %d: в OpenCV H — половина градусов" % hsv[..., 0].max())
    return img


def demo_1_2(img):
    print("\n--- Демо 1.2 — LUT: гамма за миллисекунды")
    g = np.arange(256) / 255.0
    lut = np.uint8(np.round(255 * g ** (1 / 2.2)))        # гамма-кодирование
    t0 = time.perf_counter()
    fast = cv2.LUT(img, lut)
    t1 = time.perf_counter()
    slow = np.uint8(255 * (img / 255.0) ** (1 / 2.2))
    t2 = time.perf_counter()
    print("cv2.LUT: %.1f мс, np.power: %.0f мс — в %.0f раз дольше"
          % (1e3 * (t1 - t0), 1e3 * (t2 - t1), (t2 - t1) / (t1 - t0)))
    print("код 128 → %d: половина света — не половина кода" % lut[128])
    print("совпадение до ±1 кода:", np.abs(fast.astype(int) - slow).max() <= 1)


def demo_1_3(img):
    print("\n--- Демо 1.3 — гистограмма и эквализация")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lin = (gray / 255.0) ** 2.2                          # код → свет
    dark = np.uint8(255 * (0.25 * lin) ** (1 / 2.2))     # −2 EV → код
    hist = cv2.calcHist([dark], [0], None, [256], [0, 256]).ravel()
    print("кодов < 16: %.0f %%, уровней занято: %d из 256"
          % (100 * hist[:16].sum() / hist.sum(), (hist > 0).sum()))
    eq = cv2.equalizeHist(dark)
    print("σ до %.1f → после %.1f; уровней после: %d"
          % (dark.std(), eq.std(), len(np.unique(eq))))
    return gray


def demo_1_4(gray):
    print("\n--- Демо 1.4 — CLAHE против equalizeHist")
    h, w = gray.shape
    x, x0 = np.arange(w, dtype=np.float32), 0.6 * w
    light = np.where(x < x0, 1.0, 1 - 0.85 * (x - x0) / (w - 1 - x0))
    uneven = np.uint8(gray * light.astype(np.float32))   # тень в правом углу
    shadow = (slice(None), slice(w - 300, None))         # правые 300 px
    eq = cv2.equalizeHist(uneven)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for name, im in (("вход", uneven), ("equalizeHist", eq),
                     ("CLAHE 2.0", clahe.apply(uneven))):
        print("%-13s σ в тени %5.1f · светлая часть: среднее %3.0f"
              % (name, im[shadow].std(), im[:, :int(x0)].mean()))


if __name__ == "__main__":
    img = demo_1_1()
    demo_1_2(img)
    gray = demo_1_3(img)
    demo_1_4(gray)
