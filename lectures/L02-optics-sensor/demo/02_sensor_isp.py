# -*- coding: utf-8 -*-
"""L2, демо-блок 2: сенсор и ISP — четыре фрагмента с лекции.

Запуск из корня репозитория:

    python lectures/L02-optics-sensor/demo/02_sensor_isp.py

Каждый фрагмент — ровно тот код, что стоит на слайде; вывод — те числа, что
стоят в @result. Мозаика Байера здесь имитируется из готового JPEG — фото уже
прошло чей-то демозаик, для изучения артефактов этого достаточно (обсуждение
пределов имитации — на занятии 2).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # корень репозитория

import cv2                                  # noqa: E402
import numpy as np                          # noqa: E402
from cvcourse import io as cio, metrics     # noqa: E402

PHOTO = os.path.join(HERE, "iu-building.jpg")


def demo_2_1():
    print("\n--- Демо 2.1 — мозаика и демозаик")
    img = cio.imread(PHOTO)
    mosaic = np.zeros(img.shape[:2], np.uint8)   # RGGB: R в (0,0)
    mosaic[0::2, 0::2] = img[0::2, 0::2, 2]      # R
    mosaic[0::2, 1::2] = img[0::2, 1::2, 1]      # G
    mosaic[1::2, 0::2] = img[1::2, 0::2, 1]      # G
    mosaic[1::2, 1::2] = img[1::2, 1::2, 0]      # B
    back = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR)
    print("PSNR после демозаика: %.1f дБ" % metrics.psnr(img, back))


def demo_2_2():
    print("\n--- Демо 2.2 — ложный цвет на 1 px")
    t = np.full((200, 320), 255, np.uint8)   # серая мишень: R=G=B
    t[:, ::4] = 0                            # чёрные линии в 1 px
    back = cv2.cvtColor(t, cv2.COLOR_BayerBG2BGR).astype(int)
    false = (np.abs(back[..., 2] - back[..., 0]) > 32).mean()
    print("ложный цвет: %.0f %% пикселей" % (100 * false))


def demo_2_3():
    print("\n--- Демо 2.3 — шум и стек кадров")
    rng = np.random.default_rng(0)
    N = 400                                  # фотонов на пиксель
    stack = rng.poisson(N, (16, 200, 200)).astype(np.float32)
    print("один кадр:  σ = %.1f (√N = %.1f)"
          % (stack[0].std(), N ** 0.5))
    print("среднее 16: σ = %.1f — в 4 раза меньше"
          % stack.mean(0).std())


def demo_2_4():
    print("\n--- Демо 2.4 — rolling shutter руками")
    frame = np.full((400, 600), 235, np.uint8)
    for x0 in (120, 300, 480):
        frame[:, x0:x0 + 26] = 40            # три вертикальных столба
    rs = np.stack([np.roll(frame[y], y // 4) for y in range(400)])
    print("панорама при чтении: наклон %.0f°"
          % np.degrees(np.arctan(0.25)))
    return rs


if __name__ == "__main__":
    demo_2_1()
    demo_2_2()
    demo_2_3()
    demo_2_4()
