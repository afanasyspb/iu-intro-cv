# -*- coding: utf-8 -*-
"""Автотесты занятия 4 «Геометрия на плоскости». Самопроверка студента и CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:

  1. путь в переменной окружения ``CVCOURSE_LAB_NB`` (так CI ассистента гоняет решение);
  2. ``lab04_solution.ipynb`` рядом — появляется в репозитории после занятия;
  3. ``lab04.ipynb`` — заготовка, заполненная студентом; тесты незаполненных TODO
     пропускаются, а не падают.

Данные синтетические: ни фото лекции, ни камера тестам не нужны.
"""
import io
import itertools
import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = (os.environ.get("CVCOURSE_LAB_NB"),
              os.path.join(HERE, "lab04_solution.ipynb"),
              os.path.join(HERE, "lab04.ipynb"))
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
WANTED = ("order_corners", "rectify", "upscale_table")


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab04_solution.ipynb, ни lab04.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab04", "__todo__": set()}
    exec("import cv2, numpy as np, timeit\nfrom cvcourse import metrics", g)
    exec("FLAGS = ('INTER_NEAREST', 'INTER_LINEAR', 'INTER_CUBIC', 'INTER_LANCZOS4')", g)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        hit = [w for w in WANTED if "def " + w in src]
        if not hit:
            continue
        sep = chr(10) * 3               # определение отделено пустой строкой; до него могут стоять константы
        for chunk in src.split(sep):
            exec(compile(chunk, "<lab04>", "exec"), g)
            defined = [w for w in hit if "def " + w in chunk]
            if defined:
                if "NotImplementedError" in chunk:   # пропуск в заготовке не заполнен
                    g["__todo__"].update(defined)
                break                                # дальше в ячейке — ассерты, они не нужны
    for name in WANTED:
        if name not in g:
            pytest.skip("в ноутбуке нет функции %s" % name)
    return g


def need(sol, *names):
    """Тест пропускается, если нужный TODO ещё не заполнен."""
    todo = [n for n in names if n in sol["__todo__"]]
    if todo:
        pytest.skip("не заполнен TODO: " + ", ".join(todo))


def apply_h(H, pts):
    p = np.c_[np.asarray(pts, np.float64), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


# ------------------------------------------------------------- сцены ---
QUAD = np.float32([[120, 90], [430, 60], [470, 400], [90, 380]])      # tl, tr, br, bl — выпуклый, «под углом»
PAGE = (200, 283)                                                      # документ «в лоб»: (W, H), ≈ A4


@pytest.fixture(scope="module")
def document_scene():
    """Страница с текстом и сеткой, «висящая» в кадре 640 × 480 по углам QUAD; эталон известен."""
    w, h = PAGE
    page = np.full((h, w, 3), 255, np.uint8)
    for i, t in enumerate(("Computer", "Vision", "IU 2026")):
        cv2.putText(page, t, (12, 50 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    for k in range(0, w, 40):
        cv2.line(page, (k, 200), (k, h - 1), (0, 0, 0), 2)
    cv2.rectangle(page, (2, 2), (w - 3, h - 3), (0, 0, 0), 2)
    bg = np.full((480, 640, 3), (120, 130, 140), np.uint8)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    Hp = cv2.getPerspectiveTransform(src, QUAD)
    m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Hp, (640, 480)) > 127
    scene = bg.copy()
    scene[m] = cv2.warpPerspective(page, Hp, (640, 480))[m]
    return scene, page


def ink_match(a, b):
    ink = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 128
    return 100.0 * float((ink(a) == ink(b)).mean())


@pytest.fixture(scope="module")
def smooth_photo():
    """Гладкая «фотография» 240 × 320: градиент, пятна и кольца — без текста, чтобы порядок флагов был честным."""
    yy, xx = np.mgrid[:240, :320].astype(np.float32)
    img = 90 + 60 * np.sin(xx / 23.0) * np.cos(yy / 17.0) + 0.2 * xx
    for cx, cy, r in ((80, 70, 45), (230, 150, 60), (150, 200, 30)):
        d = np.hypot(xx - cx, yy - cy)
        img += 70 * np.exp(-(d / r) ** 2) * np.cos(d / 6.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.merge([img, cv2.flip(img, 1), cv2.flip(img, 0)])


# ------------------------------------------------------------- тесты ---
def test_environment():
    assert cv2.__version__.startswith("4.14"), (
        "Курс собран на OpenCV 4.14.0.94, найдено " + cv2.__version__)
    assert hasattr(cv2, "ximgproc"), "нужен opencv-contrib-python, а не opencv-python"


def test_order_corners_all_permutations(sol):
    """Любой порядок четырёх точек → tl, tr, br, bl; результат — float32 (4, 2)."""
    need(sol, "order_corners")
    for perm in itertools.permutations(range(4)):
        got = sol["order_corners"](QUAD[list(perm)])
        assert isinstance(got, np.ndarray) and got.shape == (4, 2) and got.dtype == np.float32
        assert np.array_equal(got, QUAD), "перестановка %s дала %s" % (perm, got.tolist())


def test_order_corners_rectangle_and_input_types(sol):
    """Прямоугольник по осям; вход — список кортежей, а не только массив."""
    need(sol, "order_corners")
    got = sol["order_corners"]([(300, 50), (100, 200), (300, 200), (100, 50)])
    assert np.array_equal(got, [[100, 50], [300, 50], [300, 200], [100, 200]])
    got = sol["order_corners"](np.array([[5.5, 9.0], [1.0, 1.0], [9.0, 1.5], [1.5, 8.0]], np.float64))
    assert got.dtype == np.float32 and np.array_equal(got, [[1.0, 1.0], [9.0, 1.5], [5.5, 9.0], [1.5, 8.0]])


def test_rectify_size_and_corner_mapping(sol, document_scene):
    """С заданным size: форма (H, W), H переводит углы кадра ровно в углы листа, порядок входа не важен."""
    need(sol, "order_corners", "rectify")
    scene, page = document_scene
    front, H = sol["rectify"](scene, QUAD[[2, 0, 3, 1]], size=PAGE)
    assert front.shape[:2] == (PAGE[1], PAGE[0]), "dsize = (W, H): ожидалось %d строк × %d столбцов" % (PAGE[1], PAGE[0])
    dst = np.float32([[0, 0], [PAGE[0] - 1, 0], [PAGE[0] - 1, PAGE[1] - 1], [0, PAGE[1] - 1]])
    assert np.abs(apply_h(H, QUAD) - dst).max() < 1e-3
    same, _ = sol["rectify"](scene, QUAD, size=PAGE)
    assert np.array_equal(front, same)


def test_rectify_matches_ground_truth(sol, document_scene):
    """Выпрямленный вид совпадает с документом «в лоб» после бинаризации (две интерполяции — > 97 %)."""
    need(sol, "order_corners", "rectify")
    scene, page = document_scene
    front, _ = sol["rectify"](scene, QUAD, size=PAGE)
    m = ink_match(front, page)
    assert m > 97, "совпало только %.1f %%" % m


def test_rectify_aspect_and_default_size(sol, document_scene):
    """Без size: ширина — большая из горизонтальных сторон; aspect задаёт W / H; без aspect — по сторонам в кадре."""
    need(sol, "order_corners", "rectify")
    scene, page = document_scene
    top = np.linalg.norm(QUAD[1] - QUAD[0]); bottom = np.linalg.norm(QUAD[2] - QUAD[3])
    left = np.linalg.norm(QUAD[3] - QUAD[0]); right = np.linalg.norm(QUAD[2] - QUAD[1])
    f_a, _ = sol["rectify"](scene, QUAD, aspect=210 / 297)
    assert abs(f_a.shape[1] - max(top, bottom)) <= 1
    assert abs(f_a.shape[1] / f_a.shape[0] - 210 / 297) < 0.01
    f_p, _ = sol["rectify"](scene, QUAD)
    assert abs(f_p.shape[1] - max(top, bottom)) <= 1 and abs(f_p.shape[0] - max(left, right)) <= 1


def test_upscale_table_shape_and_direct_psnr(sol, smooth_photo):
    """Массив (флаги × [PSNR, мс]); PSNR совпадает с прямым вызовом при уменьшении INTER_AREA."""
    need(sol, "upscale_table")
    img = smooth_photo
    flags = ("INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_LANCZOS4")
    t = np.asarray(sol["upscale_table"](img, 4, flags), np.float64)
    assert t.shape == (4, 2) and np.all(np.isfinite(t))
    from cvcourse.metrics import psnr
    small = cv2.resize(img, (80, 60), interpolation=cv2.INTER_AREA)
    for i, name in enumerate(flags):
        ref = psnr(img, cv2.resize(small, (320, 240), interpolation=getattr(cv2, name)))
        assert abs(t[i, 0] - ref) < 1e-6, "строка %s: %.3f против %.3f" % (name, t[i, 0], ref)


def test_upscale_table_order_and_milliseconds(sol, smooth_photo):
    """На гладкой картинке NEAREST хуже CUBIC; время положительное и в миллисекундах; factor передаётся."""
    need(sol, "upscale_table")
    img = smooth_photo
    t4 = np.asarray(sol["upscale_table"](img, 4), np.float64)
    t8 = np.asarray(sol["upscale_table"](img, 8), np.float64)
    assert t4[0, 0] < t4[2, 0] and t8[0, 0] < t8[2, 0], "NEAREST должен проигрывать CUBIC на гладком фото"
    assert np.all(t4[:, 0] > t8[:, 0]), "каждый флаг: ×8 теряет больше, чем ×4"
    assert np.all(t4[:, 1] > 0) and 0.02 < t4[3, 1] < 100, "время — в миллисекундах, LANCZOS4 на 320 × 240 — десятые доли мс"
