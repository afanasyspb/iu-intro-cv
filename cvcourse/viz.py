# -*- coding: utf-8 -*-
"""Показ изображений — без BGR/RGB-ошибок и одинаково в Colab и локально.

``cv2.imshow`` не работает в Colab, а ``plt.imshow`` ждёт RGB и молча рисует
синие лица, если дать ему BGR. Здесь одна функция :func:`show`, которая знает
про оба подвоха: канальность определяется по форме массива, а порядок каналов
объявляется явно параметром ``bgr``.
"""
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io import IN_COLAB

#: Цвета курса в BGR — совпадают с палитрой колод.
BLUE = (192, 112, 0)      # #0070C0
ORANGE = (17, 90, 197)    # #C55A11
GREEN = (53, 130, 84)     # #548235
INK = (31, 29, 29)


def _to_rgb(img, bgr=True):
    if img is None:
        raise ValueError("show(): передан None — вероятно, imread вернул пустоту")
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        a = (a * 255).astype(np.uint8)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if a.shape[2] == 4:
        return cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA if bgr else cv2.COLOR_RGBA2RGBA)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if bgr else a


def show(img, title=None, *, bgr=True, size=6, cmap=None, axis=False):
    """Показывает одно изображение.

    ``bgr=True`` — вход в порядке OpenCV (по умолчанию, потому что почти всё
    в курсе приходит из ``cv2``). Для массива, собранного вручную в RGB,
    ставится ``bgr=False``.
    """
    rgb = _to_rgb(img, bgr)
    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(size, size * h / max(w, 1)))
    ax.imshow(rgb if cmap is None else np.asarray(img), cmap=cmap)
    if title:
        ax.set_title(title, fontsize=11)
    ax.axis("on" if axis else "off")
    fig.tight_layout()
    plt.show()
    return ax


def grid(images, titles=None, *, cols=None, bgr=True, size=4, cmap=None, suptitle=None):
    """Сетка изображений — основной способ показывать «до / после».

    ``images`` — список массивов либо словарь ``{подпись: массив}``.
    """
    if isinstance(images, dict):
        titles = list(images.keys())
        images = list(images.values())
    images = list(images)
    n = len(images)
    if n == 0:
        raise ValueError("grid(): пустой список изображений")
    cols = cols or min(n, 3)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * 0.78))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(_to_rgb(images[i], bgr) if cmap is None else images[i], cmap=cmap)
            if titles is not None and i < len(titles):
                ax.set_title(str(titles[i]), fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    plt.show()
    return axes[:n]


def draw_keypoints(img, keypoints, *, color=ORANGE, size=None, rich=True):
    """Накладывает точки интереса. ``rich=True`` рисует масштаб и ориентацию."""
    flags = (cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich
             else cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    base = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = cv2.drawKeypoints(base, list(keypoints), None, color=color, flags=flags)
    if size:
        out = cv2.resize(out, None, fx=size, fy=size, interpolation=cv2.INTER_AREA)
    return out


def draw_matches(img1, kp1, img2, kp2, matches, *, limit=50, mask=None):
    """Рисует сопоставления. ``mask`` — маска инлаеров из ``findHomography``."""
    matches = list(matches)[:limit]
    mm = None
    if mask is not None:
        mm = np.asarray(mask).ravel()[:len(matches)].astype(np.uint8).tolist()
    return cv2.drawMatches(
        img1, list(kp1), img2, list(kp2), matches, None,
        matchColor=GREEN, singlePointColor=(160, 160, 160), matchesMask=mm,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def hist(img, *, bins=256, title="Гистограмма", size=(7, 3)):
    """Гистограмма яркости или трёх каналов — L3 и ДЗ 1."""
    fig, ax = plt.subplots(figsize=size)
    a = np.asarray(img)
    if a.ndim == 2:
        ax.plot(cv2.calcHist([a], [0], None, [bins], [0, 256]), color="#1D1D1F")
    else:
        for i, c in enumerate(("#0070C0", "#548235", "#C55A11")):   # B, G, R
            ax.plot(cv2.calcHist([a], [i], None, [bins], [0, 256]), color=c, lw=1.2)
    ax.set_xlim(0, bins)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=.25)
    fig.tight_layout()
    plt.show()
    return ax


def flow_to_color(flow):
    """Оптический поток -> цветная картинка: оттенок = направление, яркость = модуль."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def imshow_window(name, img, wait=1):
    """``cv2.imshow`` с понятной ошибкой в Colab вместо зависания ядра."""
    if IN_COLAB:
        raise RuntimeError(
            "cv2.imshow не работает в Colab. Используйте cvcourse.viz.show(); "
            "интерактив (мышь, трекбары, waitKey) требует локального запуска.")
    cv2.imshow(name, img)
    return cv2.waitKey(wait)
