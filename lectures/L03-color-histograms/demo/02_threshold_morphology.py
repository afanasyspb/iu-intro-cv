# -*- coding: utf-8 -*-
"""L3, демо-блок 2: пороги, маска по цвету, морфология, связные компоненты.

Запуск из корня репозитория:

    python lectures/L03-color-histograms/demo/02_threshold_morphology.py

Каждый фрагмент — ровно тот код, что стоит на слайде; вывод — те числа, что
стоят в @result. Данные рядом со скриптом:

    text-lamp.png   страница текста при свете 1 → 0.24 слева направо (+ шум)
    text-flat.png   та же страница при ровном свете (+ шум) — для сравнения
    text-gt.png     эталон: чернила 0, бумага 255
    ball-day.jpg    фото здания ИУ + пурпурный мяч, дневной свет
    ball-lamp.jpg   тот же кадр вдвое тусклее и чуть теплее («лампа»)
    mask-noisy.png  маска мяча с крапинками снаружи и дырами внутри
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # корень репозитория

import cv2                                  # noqa: E402
import numpy as np                          # noqa: E402
from cvcourse import io as cio              # noqa: E402


def data(name):
    return os.path.join(HERE, name)


# ядра морфологии — определяются в демо 2.2, переиспользуются в 2.3
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def demo_2_1():
    print("\n--- Демо 2.1 — четыре бинаризации текста")
    gt = cio.imread(data("text-gt.png"), "gray")      # чернила 0, бумага 255
    page = cio.imread(data("text-lamp.png"), "gray")  # свет 1 → 0.24 слева направо
    t, otsu = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = [("порог 128", cv2.threshold(page, 128, 255, cv2.THRESH_BINARY)[1]),
           ("Otsu, t = %d" % t, otsu)]
    for name, flag in (("adaptive mean", cv2.ADAPTIVE_THRESH_MEAN_C),
                       ("adaptive gauss", cv2.ADAPTIVE_THRESH_GAUSSIAN_C)):
        res.append((name, cv2.adaptiveThreshold(
            page, 255, flag, cv2.THRESH_BINARY, 31, 15)))
    for name, b in res:
        print("%-15s верных пикселей %.1f %%" % (name, 100 * (b == gt).mean()))


def demo_2_2():
    print("\n--- Демо 2.2 — маска по цвету → морфология → bbox")
    frame = cio.imread(data("ball-lamp.jpg"))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (140, 121, 40), (168, 230, 255))   # H, S — с кадра «день»
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)          # крапинки прочь
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9)         # дыры (блик) зашить
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask)
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, area = stats[i]
    print("компонент: %d; bbox (%d, %d, %d × %d), площадь %d px"
          % (n - 1, x, y, w, h, area))


def demo_2_3():
    print("\n--- Демо 2.3 — крапинки, дыры и связные компоненты")
    noisy = cio.imread(data("mask-noisy.png"), "gray")   # маска + крапинки + дыры
    holes = lambda m: cv2.connectedComponents(255 - m)[0] - 2   # фон минус внешний
    n0 = cv2.connectedComponentsWithStats(noisy)[0] - 1
    opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, k5)        # ядра — из демо 2.2
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k9)
    print("компонент: %d → %d после открытия 5 × 5"
          % (n0, cv2.connectedComponentsWithStats(opened)[0] - 1))
    print("дыр: %d → %d после закрытия 9 × 9" % (holes(noisy), holes(closed)))
    print("площадь: %d → %d px (одна эрозия оставила бы %d)"
          % ((noisy > 0).sum(), (closed > 0).sum(), (cv2.erode(noisy, k5) > 0).sum()))


def demo_2_4():
    print("\n--- Демо 2.4 — где ломается hue")
    rng = np.random.default_rng(0)
    grey = rng.normal(128, 4, (100, 100, 3))                  # серый + шум σ = 4
    h = cv2.cvtColor(np.uint8(np.clip(grey, 0, 255)), cv2.COLOR_BGR2HSV)[..., 0]
    print("серый патч: hue от %d до %d, σ = %.0f — у серого нет оттенка"
          % (h.min(), h.max(), h.std()))
    red = rng.normal((30, 30, 220), 8, (100, 100, 3))         # красный + шум σ = 8
    h = cv2.cvtColor(np.uint8(np.clip(red, 0, 255)), cv2.COLOR_BGR2HSV)[..., 0]
    print("красный: hue < 10 у %.0f %% пикселей, hue > 170 у %.0f %%"
          % (100 * (h < 10).mean(), 100 * (h > 170).mean()))


if __name__ == "__main__":
    demo_2_1()
    demo_2_2()
    demo_2_3()
    demo_2_4()
