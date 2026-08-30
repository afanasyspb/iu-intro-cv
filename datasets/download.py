# -*- coding: utf-8 -*-
"""Скачать датасеты курса одной командой.

    python datasets/download.py                  # всё, что скачивается автоматически
    python datasets/download.py middlebury-motorcycle sintel-clean-alley
    python datasets/download.py --list

Кэш общий на машину: ``~/.cache/cvcourse`` либо ``$CVCOURSE_CACHE``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvcourse import datasets


def main(argv):
    if "--list" in argv:
        datasets.available()
        return 0

    names = [a for a in argv if not a.startswith("-")]
    if not names:
        names = [n for n, e in datasets.REGISTRY.items() if e["url"]]

    failed = []
    for n in names:
        print("==", n)
        try:
            print("   ->", datasets.fetch(n))
        except Exception as e:                      # noqa: BLE001
            print("   !!", e)
            failed.append(n)

    if failed:
        print("
Не скачано: %s" % ", ".join(failed))
        print("Инструкции по ручной установке — в data/README.md нужного занятия.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
