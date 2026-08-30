# -*- coding: utf-8 -*-
"""Проверка окружения: версия OpenCV и наличие contrib-модулей.

Запускается в CI и вручную студентом первой командой после установки:

    python tools/check_env.py
"""
import sys

WANT = "4.14.0"
PIP = "pip install opencv-contrib-python==4.14.0.94"


def main():
    problems = []

    if sys.version_info < (3, 10):
        problems.append("Python %d.%d — нужен 3.10 или новее"
                        % sys.version_info[:2])

    try:
        import cv2
    except ImportError:
        print("OpenCV не установлен.  " + PIP)
        return 1

    if cv2.__version__ != WANT:
        problems.append("OpenCV %s, курс собран на %s.  %s"
                        % (cv2.__version__, WANT, PIP))
    for mod, why in (("ximgproc", "WLS-фильтр, L12"), ("aruco", "ChArUco, L11")):
        if not hasattr(cv2, mod):
            problems.append("нет cv2.%s (%s) — установлен opencv-python вместо "
                            "opencv-contrib-python.  %s" % (mod, why, PIP))

    try:
        import numpy, matplotlib, sklearn, skimage, scipy  # noqa: F401
    except ImportError as e:
        problems.append("нет пакета: %s.  pip install -r requirements.txt" % e.name)

    try:
        import cvcourse
        ver = cvcourse.__version__
    except Exception as e:                        # noqa: BLE001
        problems.append("cvcourse не импортируется: %s" % e)
        ver = "?"

    if problems:
        print("Окружение не готово:")
        for p in problems:
            print("  - " + p)
        print("\nПодробно: docs/setup-guide.md")
        return 1

    print("Окружение в порядке: OpenCV %s, cvcourse %s, Python %d.%d"
          % (cv2.__version__, ver, sys.version_info[0], sys.version_info[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
