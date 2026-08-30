# -*- coding: utf-8 -*-
"""Чтение изображений и видео — одинаково локально и в Colab.

Главное, что здесь есть: ``imread`` падает с внятным сообщением вместо того,
чтобы вернуть ``None``. Молчаливый ``None`` от ``cv2.imread`` — источник
примерно половины «у меня не работает» на первых занятиях: путь неверен,
кириллица в пути, файл не скачан, — а ошибка всплывает через три ячейки
в виде ``TypeError: 'NoneType' object is not subscriptable``.
"""
import os
import sys

import cv2
import numpy as np

IN_COLAB = "google.colab" in sys.modules


# --------------------------------------------------------------- картинки ---
def imread(path, mode="color"):
    """Читает изображение и **гарантирует** результат либо внятную ошибку.

    mode: ``"color"`` -> BGR ``uint8``; ``"gray"`` -> одноканальное;
    ``"unchanged"`` -> как в файле, вместе с альфа-каналом и глубиной 16 бит.
    """
    flags = {"color": cv2.IMREAD_COLOR,
             "gray": cv2.IMREAD_GRAYSCALE,
             "unchanged": cv2.IMREAD_UNCHANGED}
    if mode not in flags:
        raise ValueError("mode должен быть color/gray/unchanged, получено %r" % mode)

    path = str(path)
    # cv2.imread на Windows не открывает пути с кириллицей — читаем через numpy
    data = _read_bytes(path)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags[mode])
    if img is None:
        raise ValueError(
            "OpenCV не смог декодировать %s — файл повреждён или это не изображение "
            "(размер %d байт)" % (path, len(data)))
    return img


def imread_rgb(path):
    """То же, что :func:`imread`, но сразу в RGB — для ``matplotlib``."""
    return cv2.cvtColor(imread(path, "color"), cv2.COLOR_BGR2RGB)


def imwrite(path, img):
    """Сохраняет изображение; создаёт каталог и работает с кириллицей в пути."""
    path = str(path)
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("OpenCV не смог закодировать изображение в %s" % ext)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return path


def _read_bytes(path):
    if not os.path.exists(path):
        hint = ""
        d = os.path.dirname(os.path.abspath(path)) or "."
        if os.path.isdir(d):
            near = sorted(os.listdir(d))[:6]
            hint = "\n  В каталоге %s есть: %s" % (d, ", ".join(near) or "(пусто)")
        else:
            hint = "\n  Каталога %s не существует." % d
        raise FileNotFoundError(
            "Файл не найден: %s%s\n  Данные качаются скриптом datasets/download.py "
            "или функцией cvcourse.datasets.fetch()." % (path, hint))
    if os.path.getsize(path) == 0:
        raise ValueError("Файл пустой: %s — скачивание оборвалось, скачайте заново" % path)
    with open(path, "rb") as f:
        return f.read()


# ------------------------------------------------------------------ видео ---
def video_frames(source, *, max_frames=None, step=1, gray=False, resize=None):
    """Генератор кадров из файла или веб-камеры.

    ``source`` — путь к файлу либо индекс камеры (``0``). ``step=3`` вернёт
    каждый третий кадр; ``resize=(640, 360)`` уменьшит кадр сразу при чтении.

    Захват всегда освобождается, даже если цикл прерван — иначе камера
    остаётся занятой до перезапуска ядра, и это второй по частоте вопрос
    на занятиях по видео.
    """
    if isinstance(source, str) and not os.path.exists(source):
        _read_bytes(source)                       # ради подробного сообщения
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            "Не удалось открыть источник видео: %r.\n"
            "  Для камеры в Colab потокового видео нет — нужен локальный запуск "
            "(WP0 §8.2, п. 4)." % source)
    try:
        i = n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                if resize is not None:
                    frame = cv2.resize(frame, tuple(resize), interpolation=cv2.INTER_AREA)
                if gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield frame
                n += 1
                if max_frames is not None and n >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def video_info(path):
    """Словарь с параметрами видеофайла: кадры, fps, размер, длительность."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео: %s" % path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {"frames": n, "fps": fps, "width": w, "height": h,
            "seconds": n / fps if fps else 0.0}


def as_float(img):
    """``uint8`` -> ``float32`` в диапазоне [0, 1]. Против молчаливого переполнения."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def as_uint8(img):
    """``float`` [0, 1] -> ``uint8`` с обрезкой. Всегда через это, а не через astype."""
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0 if img.max() <= 1.0 + 1e-6 else img, 0, 255).astype(np.uint8)
