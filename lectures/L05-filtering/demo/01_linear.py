# -*- coding: utf-8 -*-
"""L5, демо-блок 1: линейная фильтрация — четыре фрагмента с лекции.

Запуск из корня репозитория:

    python lectures/L05-filtering/demo/01_linear.py

Каждый фрагмент — ровно тот код, что стоит на слайде; вывод — те числа, что
стоят в @result. Данные рядом со скриптом:

    iu-building.jpg   фотография здания ИУ, 1920 × 1279 (читается в сером)

Шум порождается с фиксированными зёрнами — числа воспроизводимы; тайминги
демо 1.3 зависят от машины — на слайде порядок величины.
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


def demo_1_1(g):
    print("\n--- Демо 1.1 — корреляция против свёртки")
    k = np.zeros((5, 5), np.float32); k[2, 4] = 1             # «взять пиксель справа на 2»
    corr = cv2.filter2D(g, -1, k)                             # filter2D — корреляция
    conv = cv2.filter2D(g, -1, cv2.flip(k, -1))               # свёртка = перевёрнутое ядро
    ref = np.zeros_like(g); ref[:, 2:] = g[:, :-2]            # свёртка руками: сдвиг вправо
    d = lambda a, b: int(np.abs(a.astype(int) - b)[10:-10, 10:-10].max())   # noqa: E731
    print("filter2D против свёртки: %d кодов; с flip: %d" % (d(corr, ref), d(conv, ref)))
    v = cv2.getGaussianKernel(5, 1.0); gk = v @ v.T           # симметричное ядро
    print("Гаусс: %d" % d(cv2.filter2D(g, -1, gk), cv2.filter2D(g, -1, cv2.flip(gk, -1))))


def demo_1_2(g):
    print("\n--- Демо 1.2 — σ по метрике, не по правилу")
    rng = np.random.default_rng(0)
    noisy = np.clip(g + rng.normal(0, 25, g.shape), 0, 255).astype(np.uint8)
    sweep = [(psnr(g, cv2.GaussianBlur(noisy, (0, 0), s)), s)
             for s in np.arange(0.5, 6.01, 0.25)]
    best_db, best_s = max(sweep)
    print("шум σ 25: %.1f дБ → лучшая σ фильтра %g: %.1f дБ"
          % (psnr(g, noisy), best_s, best_db))
    for s in (1, 2, 4):
        print("σ = %g: %.1f дБ" % (s, psnr(g, cv2.GaussianBlur(noisy, (0, 0), s))))
    return noisy


def demo_1_3(g):
    print("\n--- Демо 1.3 — сепарабельность и время")
    ms = lambda fn: 1e3 * min(timeit.repeat(fn, number=1, repeat=5))   # noqa: E731
    for k in (5, 15, 31, 61):
        v = cv2.getGaussianKernel(k, k / 6)                   # столбец k × 1
        K = (v @ v.T).astype(np.float32)                      # полное ядро k × k
        print("k = %2d: filter2D %5.1f мс · sepFilter2D %4.1f · Gaussian %4.1f · box %4.1f"
              % (k, ms(lambda: cv2.filter2D(g, -1, K)),
                 ms(lambda: cv2.sepFilter2D(g, -1, v, v)),
                 ms(lambda: cv2.GaussianBlur(g, (k, k), k / 6)),
                 ms(lambda: cv2.blur(g, (k, k)))))


def demo_1_4(g):
    print("\n--- Демо 1.4 — граница: три режима, одна истина")
    y, x, n = 900, 1300, 200
    truth = cv2.GaussianBlur(g, (19, 19), 3)[y:y + n, x:x + n]   # видел соседей за краем
    crop = g[y:y + n, x:x + n]
    band = np.zeros((n, n), bool)                             # полоса 9 px = полядра
    band[:9] = band[-9:] = band[:, :9] = band[:, -9:] = True
    for name in ("BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT_101"):
        o = cv2.GaussianBlur(crop, (19, 19), 3, borderType=getattr(cv2, name))
        err = np.sqrt(((o.astype(float) - truth)[band] ** 2).mean())
        print("%-19s RMSE у края: %4.1f кода" % (name, err))


if __name__ == "__main__":
    g = cio.imread(data("iu-building.jpg"), "gray")            # 1920 × 1279
    demo_1_1(g)
    demo_1_2(g)
    demo_1_3(g)
    demo_1_4(g)
