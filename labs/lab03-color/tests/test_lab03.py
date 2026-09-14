# -*- coding: utf-8 -*-
"""Автотесты занятия 3 «Цвет, гистограммы, пороги, морфология». Самопроверка студента и CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:

  1. путь в переменной окружения ``CVCOURSE_LAB_NB`` (так CI ассистента гоняет решение);
  2. ``lab03_solution.ipynb`` рядом — появляется в репозитории после занятия;
  3. ``lab03.ipynb`` — заготовка, заполненная студентом; тесты незаполненных TODO
     пропускаются, а не падают.

Данные синтетические: ни фото, ни камера тестам не нужны.
"""
import io
import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = (os.environ.get("CVCOURSE_LAB_NB"),
              os.path.join(HERE, "lab03_solution.ipynb"),
              os.path.join(HERE, "lab03.ipynb"))
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
WANTED = ("color_mask", "largest_component", "sweep_adaptive")


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab03_solution.ipynb, ни lab03.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab03", "__todo__": set()}
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
        exec(compile(head, "<lab03>", "exec"), g)
    for name in WANTED:
        if name not in g:
            pytest.skip("в ноутбуке нет функции %s" % name)
    return g


def need(sol, *names):
    """Тест пропускается, если нужный TODO ещё не заполнен."""
    todo = [n for n in names if n in sol["__todo__"]]
    if todo:
        pytest.skip("не заполнен TODO: " + ", ".join(todo))


# ------------------------------------------------------------- сцены ---
def disc_mask(shape, cx, cy, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


@pytest.fixture(scope="module")
def purple_scene():
    """Серо-зелёный фон, пурпурный диск, крапинки того же цвета снаружи и дыра внутри."""
    rng = np.random.default_rng(0)
    img = np.full((240, 320, 3), (120, 140, 110), np.uint8)
    cv2.rectangle(img, (200, 20), (300, 120), (60, 160, 200), -1)
    gt = disc_mask(img.shape[:2], 100, 130, 45)
    img[gt] = (200, 40, 235)                                   # BGR: пурпурный
    for _ in range(40):                                        # крапинки 1–2 px вне диска
        x, y = int(rng.integers(0, 318)), int(rng.integers(0, 238))
        if not gt[y:y + 2, x:x + 2].any():
            img[y:y + 2, x:x + 2] = (200, 40, 235)
    img[126:134, 96:104] = (250, 250, 250)                     # блик — дыра 8 × 8 внутри
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img, gt


def recall_precision(mask, gt):
    m, tp = mask > 0, ((mask > 0) & gt).sum()
    return 100.0 * tp / gt.sum(), 100.0 * tp / max(m.sum(), 1)


PURPLE_LO, PURPLE_HI = (140, 120, 40), (170, 255, 255)


@pytest.fixture(scope="module")
def text_page():
    """Страница текста с градиентом света и её эталон (тот же приём, что в ноутбуке, меньше размером)."""
    rng = np.random.default_rng(1)
    page = np.full((240, 320), 255, np.uint8)
    for i, t in enumerate(("Computer Vision", "Innopolis 2026", "Otsu and adaptive", "thresholds")):
        cv2.putText(page, t, (12, 40 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    gt = np.where(page < 128, 0, 255).astype(np.uint8)
    ramp = np.linspace(1.0, 0.3, 320, dtype=np.float32)[None, :]
    lamp = np.clip(page * ramp + rng.normal(0, 6, page.shape), 0, 255).astype(np.uint8)
    return lamp, gt


# ------------------------------------------------------------- тесты ---
def test_environment():
    assert cv2.__version__.startswith("4.14"), (
        "Курс собран на OpenCV 4.14.0.94, найдено " + cv2.__version__)
    assert hasattr(cv2, "ximgproc"), "нужен opencv-contrib-python, а не opencv-python"


def test_color_mask_dtype_and_values(sol, purple_scene):
    """Маска — uint8 формы (h, w), значения только 0 и 255."""
    need(sol, "color_mask")
    img, gt = purple_scene
    m = sol["color_mask"](img, PURPLE_LO, PURPLE_HI)
    assert m.shape == img.shape[:2] and m.dtype == np.uint8
    assert set(np.unique(m).tolist()) <= {0, 255}


def test_color_mask_cleans_speckles_and_holes(sol, purple_scene):
    """После открытия и закрытия: одна компонента, нет дыр, диск найден почти целиком."""
    need(sol, "color_mask")
    img, gt = purple_scene
    m = sol["color_mask"](img, PURPLE_LO, PURPLE_HI)
    rec, prec = recall_precision(m, gt)
    assert rec > 90 and prec > 97, "recall %.1f, precision %.1f" % (rec, prec)
    assert cv2.connectedComponentsWithStats(m)[0] - 1 == 1, "крапинки должны исчезнуть после открытия"
    assert cv2.connectedComponents(255 - m)[0] - 2 == 0, "дыра от блика должна закрыться"


def test_color_mask_without_morphology_is_plain_inrange(sol, purple_scene):
    """open_k = close_k = 0 — ровно cv2.inRange в HSV."""
    need(sol, "color_mask")
    img, gt = purple_scene
    m = sol["color_mask"](img, PURPLE_LO, PURPLE_HI, 0, 0)
    ref = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
                      np.array(PURPLE_LO, np.uint8), np.array(PURPLE_HI, np.uint8))
    assert np.array_equal(m, ref)
    assert cv2.connectedComponentsWithStats(m)[0] - 1 > 1, "без морфологии крапинки остаются"


def test_color_mask_red_wraps_around_hue_zero(sol):
    """lo[0] > hi[0] — диапазон H через 179: красный диск с шумом ловится целиком."""
    need(sol, "color_mask")
    rng = np.random.default_rng(0)
    img = np.full((120, 160, 3), 40, np.uint8)
    cv2.circle(img, (80, 60), 35, (30, 30, 220), -1)
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    gt = disc_mask((120, 160), 80, 60, 35)
    wrap = sol["color_mask"](img, (170, 100, 40), (10, 255, 255))
    assert recall_precision(wrap, gt)[0] > 95
    one = sol["color_mask"](img, (0, 100, 40), (10, 255, 255))
    assert recall_precision(one, gt)[0] < recall_precision(wrap, gt)[0] - 20, \
        "одна маска [0, 10] должна терять заметную часть красного"


def test_largest_component_bbox_and_centroid(sol):
    """Из двух компонент — крупнейшая; рамка и центроид верны."""
    need(sol, "largest_component")
    m = np.zeros((100, 120), np.uint8)
    m[10:20, 10:20] = 255
    m[40:90, 50:110] = 255
    out = sol["largest_component"](m)
    assert out is not None
    (x, y, w, h), (cx, cy) = out
    assert (x, y, w, h) == (50, 40, 60, 50)
    assert abs(cx - 79.5) < 1e-6 and abs(cy - 64.5) < 1e-6


def test_largest_component_none_when_empty_or_small(sol):
    """Пустая маска и слишком мелкая компонента — None; min_area — параметр."""
    need(sol, "largest_component")
    assert sol["largest_component"](np.zeros((50, 50), np.uint8)) is None
    m = np.zeros((50, 50), np.uint8)
    m[5:15, 5:15] = 255
    assert sol["largest_component"](m) is None, "100 px < min_area = 200"
    assert sol["largest_component"](m, min_area=50)[0] == (5, 5, 10, 10)


def test_largest_component_on_color_mask(sol, purple_scene):
    """Связка TODO 1 → TODO 2: рамка диска."""
    need(sol, "color_mask", "largest_component")
    img, gt = purple_scene
    (x, y, w, h), (cx, cy) = sol["largest_component"](sol["color_mask"](img, PURPLE_LO, PURPLE_HI))
    assert abs(cx - 100) < 2 and abs(cy - 130) < 2
    assert 84 <= w <= 94 and 84 <= h <= 94


def test_sweep_adaptive_shape_and_range(sol, text_page):
    """Таблица (blockSize × C) в процентах."""
    need(sol, "sweep_adaptive")
    page, gt = text_page
    bs, cs = (11, 31, 91), (0, 10, 20)
    table = np.asarray(sol["sweep_adaptive"](page, gt, bs, cs), np.float64)
    assert table.shape == (3, 3)
    assert np.all((table >= 0) & (table <= 100))


def test_sweep_adaptive_matches_opencv_and_finds_plateau(sol, text_page):
    """Клетка равна прямому вызову adaptiveThreshold; C = 0 хуже C = 20; лучшая пара > 97 %."""
    need(sol, "sweep_adaptive")
    page, gt = text_page
    bs, cs = (11, 31, 91), (0, 10, 20)
    table = np.asarray(sol["sweep_adaptive"](page, gt, bs, cs), np.float64)
    ref = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
    assert abs(table[1, 1] - 100.0 * (ref == gt).mean()) < 1e-6
    assert table.max() > 97
    assert np.all(table[:, 0] < table[:, 2]), "без запаса C шум на бумаге даёт ложные чернила"
    gauss = np.asarray(sol["sweep_adaptive"](page, gt, bs, cs, cv2.ADAPTIVE_THRESH_GAUSSIAN_C), np.float64)
    assert gauss.shape == (3, 3) and not np.array_equal(gauss, table), "method должен передаваться в adaptiveThreshold"
