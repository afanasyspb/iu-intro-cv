# -*- coding: utf-8 -*-
"""L1, демо-блок 1: изображение как массив — четыре фрагмента с лекции.

Запуск из корня репозитория:

    python lectures/L01-intro/demo/01_image_as_array.py

Каждый фрагмент — ровно тот код, что стоит на слайде; вывод — те числа, что
стоят в @result. Если числа разошлись, правится слайд, а не скрипт.
Тайминги (демо 1.4) зависят от машины: на слайде порядок величины, не точность.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # корень репозитория

import cv2                      # noqa: E402
import numpy as np              # noqa: E402
from cvcourse import io as cio  # noqa: E402

PHOTO = os.path.join(HERE, "iu-building.jpg")


def demo_1_1():
    print("\n--- Демо 1.1 — первое изображение")
    img = cio.imread(PHOTO)           # массив, не «картинка»
    print("shape  ", img.shape)       # (строки H, столбцы W, каналы C)
    print("dtype  ", img.dtype)       # тип одного значения
    print("min/max", img.min(), img.max())
    print("bytes  ", img.nbytes)      # H · W · C · 1 байт
    return img


def demo_1_2(img):
    print("\n--- Демо 1.2 — ловушка BGR в числах")
    y, x = 897, 857                # строка y, потом столбец x
    b, g, r = img[y, x]            # пиксель оранжевого фасада
    print("канал 0 =", b, " канал 1 =", g, " канал 2 =", r)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
    print("тот же пиксель после cvtColor:", rgb[y, x])


def demo_1_3(img):
    print("\n--- Демо 1.3 — uint8 переполняется молча")
    a = np.uint8([[200]])                     # «картинка» 1 × 1
    print((a + np.uint8(100)).ravel())        # NumPy: по модулю 256
    print((a - np.uint8(210)).ravel())        # минус стал плюсом
    print(cv2.add(a, np.uint8([[100]])).ravel())  # OpenCV: насыщение

    bright = img.astype(np.float32) * 1.5     # считаем во float32…
    bright = np.clip(bright, 0, 255).astype(np.uint8)  # …и обрезаем
    print("bright:", bright.dtype, "max", bright.max())


def demo_1_4(img):
    print("\n--- Демо 1.4 — цикл против NumPy")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # один канал
    t0 = time.perf_counter()
    s = 0
    for y in range(g.shape[0]):        # 2.45 млн итераций на Python
        for x in range(g.shape[1]):
            s += int(g[y, x])
    t1 = time.perf_counter()
    m = g.mean()                       # то же самое в NumPy
    t2 = time.perf_counter()
    print("цикл  %.0f мс" % ((t1 - t0) * 1e3))
    print("NumPy %.2f мс" % ((t2 - t1) * 1e3))
    print("ответ одинаков:", abs(s / g.size - m) < 1e-6)


if __name__ == "__main__":
    image = demo_1_1()
    demo_1_2(image)
    demo_1_3(image)
    demo_1_4(image)
