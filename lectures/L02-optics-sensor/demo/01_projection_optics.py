# -*- coding: utf-8 -*-
"""L2, демо-блок 1: оптика и проекция — четыре фрагмента с лекции.

Запуск из корня репозитория:

    python lectures/L02-optics-sensor/demo/01_projection_optics.py

Каждый фрагмент — ровно тот код, что стоит на слайде; вывод — те числа, что
стоят в @result. Если числа разошлись, правится слайд, а не скрипт.
"""
import numpy as np

FX, CX, CY = 1663.0, 960.0, 540.0      # камера L1: 60°, 1920 × 1080


def demo_1_1():
    print("\n--- Демо 1.1 — проекция куба")
    fx, cx, cy = FX, CX, CY
    cube = np.array([[x, y, z] for x in (-.5, .5)
                     for y in (-.5, .5) for z in (0, 1.0)])
    for Z in (5.0, 15.0):
        p = cube + [0, 0, Z]
        u = fx * p[:, 0] / p[:, 2] + cx
        v = fx * p[:, 1] / p[:, 2] + cy
        print("куб 1 м на Z = %2.0f м: ширина %3.0f px"
              % (Z, u.max() - u.min()))


def demo_1_2():
    print("\n--- Демо 1.2 — fx из поля зрения")
    W, fov = 1920, np.deg2rad(60.0)
    fx = W / (2 * np.tan(fov / 2))
    print("fx = %.0f px (%.2f мрад на пиксель)"
          % (fx, 1000 * fov / W))
    for Z in (10, 20, 42):
        print("пешеход 1.7 м на %2d м: %3.0f px" % (Z, fx * 1.7 / Z))


def demo_1_3():
    print("\n--- Демо 1.3 — дисторсия в пикселях")
    fx, cx, cy = FX, CX, CY
    k1 = -0.05                      # лёгкая «бочка», как у веб-камеры
    pts = np.array([[960.0, 540.0], [1910.0, 1070.0]])
    n = (pts - [cx, cy]) / fx       # в нормированные координаты
    r2 = (n ** 2).sum(1, keepdims=True)
    shift = np.linalg.norm(n * k1 * r2 * fx, axis=1)
    print("сдвиг: центр %.1f px, угол кадра %.1f px" % (*shift,))


def demo_1_4():
    print("\n--- Демо 1.4 — калькулятор ГРИП")
    f, c, s = 50e-3, 30e-6, 3.0    # 50 мм, допуск 30 мкм, фокус 3 м
    for N in (2, 8):
        H = f * f / (N * c) + f    # гиперфокальное расстояние
        near = H * s / (H + s - f)
        far = H * s / (H - s)
        print("f/%d: резко %.2f…%.2f м (ГРИП %.2f м)"
              % (N, near, far, far - near))


if __name__ == "__main__":
    demo_1_1()
    demo_1_2()
    demo_1_3()
    demo_1_4()
