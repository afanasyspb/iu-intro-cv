# -*- coding: utf-8 -*-
"""Заготовка занятия порождается из решения — один источник, расходиться нечему.

В решении ``labNN_solution.ipynb`` код каждого TODO обёрнут маркерами:

    # >>> SOL TODO 2 · 6–10 строк: два цикла по рядам и столбцам
    ...решение...
    # <<< SOL

Текст после ``SOL`` — подпись пропуска: до « · » идёт метка (она же в
``NotImplementedError``), после — подсказка студенту в одну строку. Скрипт
заменяет блок на заглушку с этой подсказкой, чистит выводы ячеек и пишет
``labNN.ipynb`` рядом:

    python tools/make_starter.py labs/lab01-images/lab01_solution.ipynb
    python tools/make_starter.py labs/*/lab*_solution.ipynb      # все занятия

Решения живут в репозитории ассистента ``iu-intro-cv-ta`` (клон рядом), заготовки —
здесь; ``--dest labs`` кладёт заготовку в ``labs/<каталог занятия>/`` этого репозитория:

    python tools/make_starter.py ../iu-intro-cv-ta/labs/lab01-images/lab01_solution.ipynb --dest labs

Проверка результата — ``python tools/check_starters.py``: заготовка обязана
падать ровно на ``NotImplementedError`` первого TODO.
"""
import glob
import io
import json
import os
import re
import sys

BEGIN = re.compile(r"^(?P<indent>\s*)# >>> SOL\s*(?P<label>[^·\n]*?)(?:\s*·\s*(?P<hint>.*))?\s*$")
END = re.compile(r"^\s*# <<< SOL\s*$")
BAR = "─"


def stub(indent, label, hint):
    label = label.strip() or "TODO"
    width = 63 - len(indent)
    head = (" ВАШ КОД ").center(width, BAR)
    lines = [indent + "# " + head]
    if hint:
        lines.append(indent + "# " + hint.strip())
    lines.append(indent + 'raise NotImplementedError("%s")' % label)
    lines.append(indent + "# " + BAR * width)
    return lines


def convert_source(src):
    out, k, i = [], 0, 0
    lines = src.split("\n")
    while i < len(lines):
        m = BEGIN.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not END.match(lines[j]):
            j += 1
        if j == len(lines):
            raise ValueError("маркер '# >>> SOL' без '# <<< SOL'")
        out += stub(m.group("indent"), m.group("label"), m.group("hint"))
        k += 1
        i = j + 1
    return "\n".join(out), k


def make_starter(solution_path, dest=None):
    """``dest`` — корень ``labs/`` другого репозитория: заготовка ложится
    в ``dest/<каталог занятия>/labNN.ipynb`` (решение живёт в iu-intro-cv-ta,
    заготовка — в публичном)."""
    nb = json.load(io.open(solution_path, encoding="utf-8"))
    total = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src, k = convert_source("".join(cell["source"]))
        total += k
        cell["source"] = [l + "\n" for l in src.split("\n")]
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
        cell["outputs"] = []
        cell["execution_count"] = None
    if not total:
        raise ValueError("в %s нет ни одного блока '# >>> SOL … # <<< SOL'" % solution_path)
    name = os.path.basename(solution_path).replace("_solution.ipynb", ".ipynb")
    if name == os.path.basename(solution_path):
        raise ValueError("ожидался файл вида labNN_solution.ipynb: %s" % solution_path)
    if dest:
        lab = os.path.basename(os.path.dirname(os.path.abspath(solution_path)))
        dst = os.path.join(dest, lab, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    else:
        dst = os.path.join(os.path.dirname(solution_path), name)
    with io.open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return dst, total


def main(argv):
    dest = None
    if "--dest" in argv:
        k = argv.index("--dest")
        dest = argv[k + 1]
        argv = argv[:k] + argv[k + 2:]
    paths = [p for a in argv for p in sorted(glob.glob(a))]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        dst, n = make_starter(p, dest)
        try:
            shown = os.path.relpath(dst)
        except ValueError:                  # Windows: другой диск
            shown = dst
        print("  -> %s  (%d TODO)" % (shown, n))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
