# -*- coding: utf-8 -*-
"""L5, демо-блок 2: нелинейные фильтры, поиск шаблона, метрики.

Запуск из корня репозитория:

    python lectures/L05-filtering/demo/02_nonlinear.py

Каждый фрагмент — ровно тот код, что стоит на слайде; вывод — те числа, что
стоят в @result. Данные рядом со скриптом:

    iu-building.jpg   фотография здания ИУ, 1920 × 1279 (читается в сером)

Шум порождается с теми же зёрнами, что в демо-блоке 1 и на слайдах; время
``matchTemplate`` зависит от машины — на слайде порядок величины.
"""
import os
import sys
import timeit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # корень репозитория

import cv2                                  # noqa: E402
import numpy as np                          # noqa: E402
from cvcourse import io as cio              # noqa: E402
from cvcourse.metrics import psnr, ssim     # noqa: E402


def data(name):
    return os.path.join(HERE, name)


def demo_2_1(g, noisy):
    print("\n--- Демо 2.1 — таблица «шум × фильтр»")
    poisson = np.clip(np.random.default_rng(1).poisson(g / 4.0) * 4, 0, 255).astype(np.uint8)
    u = np.random.default_rng(2).random(g.shape); sp = g.copy()
    sp[u < 0.025] = 0; sp[u > 0.975] = 255                    # соль и перец 5 %
    noises = {"Гаусс σ 25": noisy, "Пуассон": poisson, "соль-перец": sp}
    filters = {"box 5": lambda x: cv2.blur(x, (5, 5)),
               "Гаусс 1.5": lambda x: cv2.GaussianBlur(x, (0, 0), 1.5),
               "медиана 5": lambda x: cv2.medianBlur(x, 5),
               "билатеральный": lambda x: cv2.bilateralFilter(x, 9, 75, 75)}
    for n, img in noises.items():
        print("%-11s %.1f |" % (n, psnr(g, img)),
              " ".join("%s %.1f" % (f, psnr(g, fn(img))) for f, fn in filters.items()))


def demo_2_2(g, noisy):
    print("\n--- Демо 2.2 — билатеральный: что делает sigmaColor")
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.2), 60, 150),
                       np.ones((5, 5), np.uint8)) > 0                   # полоса 5 px вокруг краёв
    psnr_m = lambda a, b, m: 10 * np.log10(255 ** 2 / np.mean((a[m].astype(float) - b[m]) ** 2))   # noqa: E731
    for sc in (10, 30, 75, 150):
        o = cv2.bilateralFilter(noisy, 9, sc, 75)
        print("sigmaColor %3d: кадр %.1f · края %.1f · ровное %.1f дБ"
              % (sc, psnr(g, o), psnr_m(g, o, edges), psnr_m(g, o, ~edges)))
    o = cv2.GaussianBlur(noisy, (0, 0), 1.5)
    print("Гаусс σ 1.5:    кадр %.1f · края %.1f · ровное %.1f дБ"
          % (psnr(g, o), psnr_m(g, o, edges), psnr_m(g, o, ~edges)))


def demo_2_3(g):
    print("\n--- Демо 2.3 — matchTemplate при другом свете")
    tpl = g[960:1056, 560:656]                                # люди на лестнице, 96 × 96
    d0 = np.clip(g * 0.6 + 30, 0, 255).astype(np.uint8)       # другой свет
    dark = np.clip(d0 + np.random.default_rng(9).normal(0, 5, g.shape), 0, 255).astype(np.uint8)
    for name in ("TM_CCORR", "TM_SQDIFF", "TM_CCORR_NORMED", "TM_CCOEFF_NORMED"):
        r = cv2.matchTemplate(dark, tpl, getattr(cv2, name))
        mn, mx, lmn, lmx = cv2.minMaxLoc(r)                   # SQDIFF — минимум, остальные — максимум
        loc, peak = (lmn, mn) if "SQDIFF" in name else (lmx, mx)
        print("%-17s пик %10.3g · ошибка %4.0f px"
              % (name, peak, np.hypot(loc[0] - 560, loc[1] - 960)))
    # строка сверх слайда: время того же вызова — на слайде порядок величины
    t = 1e3 * min(timeit.repeat(lambda: cv2.matchTemplate(dark, tpl, cv2.TM_CCOEFF_NORMED), number=1, repeat=3))
    print("TM_CCOEFF_NORMED на 1920 × 1279 с шаблоном 96 × 96: %.0f мс" % t)


def demo_2_4(g):
    print("\n--- Демо 2.4 — PSNR против SSIM")
    cases = {"+20 к яркости": np.clip(g.astype(int) + 20, 0, 255).astype(np.uint8),
             "сдвиг на 1 px": np.roll(g, 1, axis=1),
             "Гаусс σ = 1":   cv2.GaussianBlur(g, (0, 0), 1.0),
             "шум σ = 10":    np.clip(g + np.random.default_rng(13).normal(0, 10, g.shape),
                                      0, 255).astype(np.uint8)}
    for name, im in cases.items():
        print("%-14s PSNR %.1f дБ · SSIM %.3f" % (name, psnr(g, im), ssim(g, im)))


if __name__ == "__main__":
    g = cio.imread(data("iu-building.jpg"), "gray")            # 1920 × 1279
    noisy = np.clip(g + np.random.default_rng(0).normal(0, 25, g.shape), 0, 255).astype(np.uint8)   # как в демо 1.2
    demo_2_1(g, noisy)
    demo_2_2(g, noisy)
    demo_2_3(g)
    demo_2_4(g)
