# -*- coding: utf-8 -*-
"""Проверяет, что выданная студенту заготовка доходит до первого TODO.

Заготовка обязана падать — но **ровно** на ``NotImplementedError`` в ячейке TODO,
а не раньше: на импорте, чтении файла или опечатке. Разница принципиальна.
В первом случае занятие начинается со второй минуты, во втором пятнадцать минут
уходит на то, чтобы понять, что виноват не студент.
"""
import glob
import os
import subprocess
import sys

OK_ERRORS = ("NotImplementedError",)
TIMEOUT = "300"


def check(path):
    """Возвращает None, если всё правильно, иначе строку с причиной."""
    r = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--stdout", "--ExecutePreprocessor.timeout=" + TIMEOUT, path],
        capture_output=True, text=True,
        env=dict(os.environ, MPLBACKEND="Agg"))
    if r.returncode == 0:
        return "выполнилась целиком — в заготовке не осталось TODO"
    if any(e in r.stderr for e in OK_ERRORS):
        return None
    tail = [l for l in r.stderr.strip().splitlines() if l.strip()]
    return "упала не на TODO: " + (tail[-1][:150] if tail else "причина неизвестна")


def main():
    starters = [p for p in sorted(glob.glob("labs/*/lab*.ipynb"))
                if "_solution" not in p and "_template" not in p]
    if not starters:
        print("заготовок не найдено — нечего проверять")
        return 0
    bad = 0
    for p in starters:
        why = check(p)
        print(("  ok    " if why is None else "  FAIL  ") + p
              + ("" if why is None else " — " + why))
        bad += why is not None
    print("проверено %d, проблем %d" % (len(starters), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
