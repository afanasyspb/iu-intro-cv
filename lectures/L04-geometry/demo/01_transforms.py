# -*- coding: utf-8 -*-
"""L4, демо-блок 1: преобразования плоскости — четыре фрагмента с лекции.

Запуск из корня репозитория:

    python lectures/L04-geometry/demo/01_transforms.py

Каждый фрагмент — ровно тот код, что стоит на слайде; вывод — те числа, что
стоят в @result. Данные рядом со скриптом:

    iu-building.jpg   фотография здания ИУ, 1920 × 1279
    poster-angle.jpg  тот же кадр с плакатом 640 × 905, «висящим» на фасаде под
                      углом; углы плаката в кадре известны точно (см. src в демо 1.2)
    poster-gt.png     плакат «в лоб» — эталон для проверки выпрямления

Тайминги демо 1.4 зависят от машины — на слайде порядок величины.
"""
import os
import sys
import timeit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))   # корень репозитория

import cv2                                  # noqa: E402
import numpy as np                          # noqa: E402
from cvcourse import io as cio              # noqa: E402


def data(name):
    return os.path.join(HERE, name)


def demo_1_1():
    print("\n--- Демо 1.1 — поворот с сохранением кадра")
    img = cio.imread(data("iu-building.jpg"))                # 1920 × 1279
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)   # 2 × 3: T(c)·R·T(−c)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = round(w * cos + h * sin), round(w * sin + h * cos)
    M[0, 2] += nw / 2 - w / 2                              # новый центр холста
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, M, (nw, nh), borderValue=(255, 255, 255))
    print("кадр %d × %d → холст %d × %d" % (w, h, nw, nh))
    print("M =", np.round(M, 2).tolist())
    assert out.shape[:2] == (nh, nw)
    return img


def demo_1_2():
    print("\n--- Демо 1.2 — четыре точки → фронтальный вид")
    img = cio.imread(data("poster-angle.jpg"))
    src = np.float32([[560, 330], [1250, 220], [1290, 1010], [600, 840]])
    W, Hh = 640, 905                                   # tl, tr, br, bl; A4 1:1.414
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]])
    H = cv2.getPerspectiveTransform(src, dst)          # 3 × 3, кадр → лист
    front = cv2.warpPerspective(img, H, (W, Hh), flags=cv2.INTER_LINEAR)
    gt = cio.imread(data("poster-gt.png"))             # плакат «в лоб»
    ink = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 128   # чернила
    p0 = cv2.perspectiveTransform(src[None, :1], H)[0, 0].round(4)
    print("H =", np.round(H, 4).tolist())
    print("угол (560, 330) →", p0)
    print("совпало: %.1f %% пикселей" % (100 * (ink(front) == ink(gt)).mean()))
    return H


def demo_1_3(H):
    print("\n--- Демо 1.3 — что сохраняется, а что нет")
    pts = np.float32([[[20, 870], [120, 870], [320, 870], [620, 870]]])   # линейка
    q = cv2.perspectiveTransform(pts, np.linalg.inv(H))[0]  # те же точки в кадре
    p = pts[0]
    d = lambda a, b: float(np.linalg.norm(a - b))
    cr = lambda u: d(u[0], u[2]) * d(u[1], u[3]) / (d(u[1], u[2]) * d(u[0], u[3]))
    print("AB/BC: лист %.3f, кадр %.3f" % (d(p[0], p[1]) / d(p[1], p[2]),
                                           d(q[0], q[1]) / d(q[1], q[2])))
    print("двойное отношение: лист %.4f, кадр %.4f" % (cr(p), cr(q)))
    A = np.float32([[1.2, 0.3, 10], [0.1, 0.8, 5]])          # аффинное 2 × 3
    sq = np.float32([[[0, 0], [100, 0], [100, 100], [0, 100]]])
    print("det A = %.2f; площадь 10000 → %.0f"
          % (np.linalg.det(A[:, :2]), cv2.contourArea(cv2.transform(sq, A))))


def demo_1_4(img):
    print("\n--- Демо 1.4 — цена варпа")
    h, w = img.shape[:2]
    ms = lambda fn: 1e3 * min(timeit.repeat(fn, number=1, repeat=5))
    R = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
    P = np.vstack([R, [0, 0, 1]]); P[2, :2] = (1e-5, 2e-5)   # чуть перспективы
    wp = lambda **kw: cv2.warpPerspective(img, P, (w, h), **kw)
    for name, fn in (("warpAffine", lambda: cv2.warpAffine(img, R, (w, h))),
                     ("warpPerspective", wp),
                     ("+ CUBIC", lambda: wp(flags=cv2.INTER_CUBIC)),
                     ("+ LANCZOS4", lambda: wp(flags=cv2.INTER_LANCZOS4))):
        print("%-16s %5.1f мс на 1920 × 1279" % (name, ms(fn)))


if __name__ == "__main__":
    img = demo_1_1()
    H = demo_1_2()
    demo_1_3(H)
    demo_1_4(img)
