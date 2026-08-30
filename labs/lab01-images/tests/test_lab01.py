# -*- coding: utf-8 -*-
"""Автотесты занятия 1. Самопроверка студента и проверка в CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:

  1. путь в переменной окружения ``CVCOURSE_LAB_NB`` (так CI ассистента гоняет решение);
  2. ``lab01_solution.ipynb`` рядом — появляется в репозитории после занятия;
  3. ``lab01.ipynb`` — заготовка, заполненная студентом; тесты незаполненных TODO
     пропускаются, а не падают.
"""
import io
import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = (os.environ.get("CVCOURSE_LAB_NB"),
              os.path.join(HERE, "lab01_solution.ipynb"),
              os.path.join(HERE, "lab01.ipynb"))
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
WANTED = ("make_scene", "describe", "cut_patches", "contact_sheet")


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab01_solution.ipynb, ни lab01.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab01", "__todo__": set()}
    exec("import cv2, numpy as np", g)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        hit = [w for w in WANTED if "def " + w in src]
        if not hit:
            continue
        sep = chr(10) * 3               # определение отделено пустой строкой
        head = src.split(sep)[0]
        if "NotImplementedError" in head:   # пропуск в заготовке не заполнен
            g["__todo__"].update(hit)
        exec(compile(head, "<lab01>", "exec"), g)
    for name in WANTED:
        if name not in g:
            pytest.skip("в ноутбуке нет функции %s" % name)
    return g


def need(sol, *names):
    """Тест пропускается, если нужный TODO ещё не заполнен."""
    todo = [n for n in names if n in sol["__todo__"]]
    if todo:
        pytest.skip("не заполнен TODO: " + ", ".join(todo))


@pytest.fixture(scope="module")
def img(sol):
    need(sol, "make_scene")
    return sol["make_scene"](640, 440, seed=1)


def test_describe_fields(sol, img):
    need(sol, "describe")
    d = sol["describe"](img)
    assert set(d) == {"h", "w", "channels", "dtype", "min", "max", "mean"}
    assert (d["h"], d["w"], d["channels"]) == (440, 640, 3)
    assert d["dtype"] == "uint8"
    assert 0 <= d["min"] <= d["mean"] <= d["max"] <= 255


def test_describe_handles_grayscale(sol, img):
    """У одноканального массива всего две оси — функция не должна падать."""
    need(sol, "describe")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert sol["describe"](g)["channels"] == 1


def test_patches_are_copies_not_views(sol, img):
    """Главный урок занятия: срез — это вид, а патч обязан быть копией."""
    need(sol, "cut_patches")
    before = img.copy()
    patches = sol["cut_patches"](img, 4, 6)
    patches[0][:] = 0
    assert np.array_equal(img, before), "патч оказался видом на исходный массив"


def test_patches_cover_the_frame(sol, img):
    """При делении с остатком не должен теряться правый и нижний край."""
    need(sol, "cut_patches")
    rows, cols = 4, 6
    patches = sol["cut_patches"](img, rows, cols)
    assert len(patches) == rows * cols
    assert sum(p.shape[0] for p in patches[::cols]) == img.shape[0]
    assert sum(p.shape[1] for p in patches[:cols]) == img.shape[1]


def test_contact_sheet_shape_and_order(sol, img):
    need(sol, "cut_patches", "contact_sheet")
    patches = sol["cut_patches"](img, 4, 6)
    sheet = sol["contact_sheet"](patches, cell=(96, 96), cols=6)
    assert sheet.shape == (4 * 96, 6 * 96, 3)
    assert sheet.dtype == np.uint8
    means = sorted(p.mean() for p in patches)
    assert means == sorted(means)


def test_area_wins_on_strong_downscale(sol, img):
    """Проверка того же утверждения, что студент видит в таблице занятия."""
    need(sol, "cut_patches")
    from cvcourse import metrics
    patches = sol["cut_patches"](img, 4, 6)
    res = {}
    for name, flag in (("NEAREST", cv2.INTER_NEAREST), ("LINEAR", cv2.INTER_LINEAR),
                       ("AREA", cv2.INTER_AREA)):
        vals = []
        for p in patches:
            down = cv2.resize(p, (24, 24), interpolation=flag)
            back = cv2.resize(down, p.shape[1::-1], interpolation=cv2.INTER_LINEAR)
            vals.append(metrics.psnr(p, back))
        res[name] = float(np.mean(vals))
    assert max(res, key=res.get) == "AREA"
    assert min(res, key=res.get) == "NEAREST"
