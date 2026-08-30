# -*- coding: utf-8 -*-
"""cvcourse — общие утилиты курса «Введение в компьютерное зрение», ИУ.

Пакет решает четыре задачи, которые иначе съедают время на каждом занятии:

* одинаковое чтение данных локально и в Colab      -> :mod:`cvcourse.io`
* показ картинок без BGR/RGB-ошибок                -> :mod:`cvcourse.viz`
* метрики, посчитанные одинаково у всех            -> :mod:`cvcourse.metrics`
* датасеты с кэшем и контрольной суммой            -> :mod:`cvcourse.datasets`

Версия OpenCV закреплена (WP0 §8.1). Проверка при импорте — не придирка:
эталонные числа в ноутбуках и автотестах привязаны к конкретной сборке,
а в OpenCV 5.0 у ``HoughLinesP`` изменилась форма результата.
"""
import sys
import warnings

__version__ = "1.0"

#: Версия, на которой собран весь курс. Менять только вместе с WP0 §8.1.
REQUIRED_OPENCV = "4.14.0"
MIN_PYTHON = (3, 10)


def _check_env():
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            "Нужен Python %d.%d или новее, найден %d.%d. См. docs/setup-guide.md"
            % (MIN_PYTHON + sys.version_info[:2]))
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "OpenCV не установлен. Выполните:\n"
            "    pip install opencv-contrib-python==%s.94\n"
            "См. docs/setup-guide.md" % REQUIRED_OPENCV)

    have = cv2.__version__
    if have != REQUIRED_OPENCV:
        major = have.split(".")[0]
        extra = ""
        if major != REQUIRED_OPENCV.split(".")[0]:
            extra = ("\n  Это другая мажорная ветка: часть вызовов курса вернёт"
                     "\n  результат другой формы (например, cv2.HoughLinesP).")
        warnings.warn(
            "Версия OpenCV %s, курс собран на %s.%s"
            "\n  Числа в эталонных результатах могут не совпасть."
            "\n  Исправить:  pip install opencv-contrib-python==%s.94"
            % (have, REQUIRED_OPENCV, extra, REQUIRED_OPENCV),
            RuntimeWarning, stacklevel=3)

    if not hasattr(cv2, "ximgproc"):
        warnings.warn(
            "Установлен opencv-python вместо opencv-contrib-python."
            "\n  Не будет ximgproc (WLS-фильтр, L12) и aruco (ChArUco, L11)."
            "\n  Исправить:  pip uninstall -y opencv-python && "
            "pip install opencv-contrib-python==%s.94" % REQUIRED_OPENCV,
            RuntimeWarning, stacklevel=3)


_check_env()

from . import datasets, io, metrics, viz  # noqa: E402
from .io import imread, imread_rgb, video_frames  # noqa: E402,F401
from .viz import show, grid, draw_keypoints, draw_matches  # noqa: E402,F401

__all__ = ["io", "viz", "metrics", "datasets",
           "imread", "imread_rgb", "video_frames",
           "show", "grid", "draw_keypoints", "draw_matches",
           "IN_COLAB", "REQUIRED_OPENCV"]

from .io import IN_COLAB  # noqa: E402,F401
