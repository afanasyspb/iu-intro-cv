# -*- coding: utf-8 -*-
"""Автотесты занятия 5 «Свёртка и фильтры». Самопроверка студента и CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:

  1. путь в переменной окружения ``CVCOURSE_LAB_NB`` (так CI ассистента гоняет решение);
  2. ``lab05_solution.ipynb`` рядом — появляется в репозитории после занятия;
  3. ``lab05.ipynb`` — заготовка, заполненная студентом; тесты незаполненных TODO
     пропускаются, а не падают.

Данные синтетические: ни фото лекции, ни сеть тестам не нужны.
"""
import io
import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = (os.environ.get("CVCOURSE_LAB_NB"),
              os.path.join(HERE, "lab05_solution.ipynb"),
              os.path.join(HERE, "lab05.ipynb"))
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
WANTED = ("conv2d", "gauss_sep", "noise_filter_table")


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab05_solution.ipynb, ни lab05.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab05", "__todo__": set()}
    exec("import cv2, numpy as np, timeit\nfrom cvcourse import metrics", g)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        hit = [w for w in WANTED if "def " + w + "(" in src]
        if not hit:
            continue
        sep = chr(10) * 3               # определение отделено пустой строкой; до него могут стоять константы
        for chunk in src.split(sep):
            exec(compile(chunk, "<lab05>", "exec"), g)
            defined = [w for w in hit if "def " + w + "(" in chunk]
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


def filter2d(img, kernel, border=cv2.BORDER_REFLECT_101):
    """Корреляция OpenCV в float64 — эталон; свёртка получается с перевёрнутым ядром."""
    return cv2.filter2D(np.float64(img), cv2.CV_64F, np.float64(kernel), borderType=border)


# ------------------------------------------------------------- сцены ---
@pytest.fixture(scope="module")
def scene():
    """«Фотография» 120 × 160: плавный фон, три круга с резким краем, полосы — есть и края, и ровные области."""
    yy, xx = np.mgrid[:120, :160].astype(np.float32)
    img = 110 + 50 * np.sin(xx / 19.0) * np.cos(yy / 13.0) + 0.2 * xx
    for cx, cy, r in ((40, 35, 22), (115, 80, 30), (70, 95, 14)):
        img[np.hypot(xx - cx, yy - cy) < r] += 60
    img[20:100:16, 100:150] -= 70
    return np.uint8(np.clip(img, 0, 255))


@pytest.fixture(scope="module")
def noises(scene):
    """Гауссов шум σ 25 и соль-перец 5 % — с фиксированными зёрнами."""
    gauss = np.clip(scene + np.random.default_rng(0).normal(0, 25, scene.shape), 0, 255).astype(np.uint8)
    u = np.random.default_rng(2).random(scene.shape)
    sp = scene.copy()
    sp[u < 0.025] = 0
    sp[u > 0.975] = 255
    return {"gauss": gauss, "sp": sp}


FILTERS = {"box 5": lambda x: cv2.blur(x, (5, 5)),
           "gauss 1.5": lambda x: cv2.GaussianBlur(x, (0, 0), 1.5),
           "median 5": lambda x: cv2.medianBlur(x, 5),
           "bilateral": lambda x: cv2.bilateralFilter(x, 9, 75, 75)}
K_ASYM = np.float64([[1, 2, 0], [0, 3, -1], [-2, 0, 1]])


# ------------------------------------------------------------- тесты ---
def test_environment():
    assert cv2.__version__.startswith("4.14"), (
        "Курс собран на OpenCV 4.14.0.94, найдено " + cv2.__version__)
    assert hasattr(cv2, "ximgproc"), "нужен opencv-contrib-python, а не opencv-python"


def test_conv2d_matches_filter2d_with_flipped_kernel(sol, scene):
    """Свёртка = filter2D с перевёрнутым ядром, до 1e-6 по всему кадру; результат — того же размера."""
    need(sol, "conv2d")
    img = scene[:48, :64]
    got = np.asarray(sol["conv2d"](img, K_ASYM), np.float64)
    assert got.shape == img.shape
    assert np.abs(got - filter2d(img, cv2.flip(K_ASYM, -1))).max() < 1e-6
    k5 = np.random.default_rng(5).normal(0, 1, (5, 5))
    assert np.abs(sol["conv2d"](img, k5) - filter2d(img, cv2.flip(k5, -1))).max() < 1e-6


def test_conv2d_impulse_response_is_the_kernel(sol):
    """Свёртка единичного импульса с ядром — само ядро; у корреляции получилось бы перевёрнутое."""
    need(sol, "conv2d")
    imp = np.zeros((9, 11))
    imp[4, 5] = 1.0
    got = np.asarray(sol["conv2d"](imp, K_ASYM), np.float64)
    assert np.allclose(got[3:6, 4:7], K_ASYM, atol=1e-9), "получилась корреляция: ядро не перевёрнуто"
    assert abs(got.sum() - K_ASYM.sum()) < 1e-9


def test_conv2d_rectangular_kernel_and_identity(sol, scene):
    """Ядро 3 × 7 (kh ≠ kw) и граница REFLECT_101; ядро-единица возвращает кадр."""
    need(sol, "conv2d")
    img = scene[10:50, 20:70]
    k = np.arange(21, dtype=np.float64).reshape(3, 7) - 8
    assert np.abs(sol["conv2d"](img, k) - filter2d(img, cv2.flip(k, -1))).max() < 1e-6
    one = np.zeros((3, 3))
    one[1, 1] = 1
    assert np.array_equal(np.asarray(sol["conv2d"](img, one), np.float64), img.astype(np.float64))


def test_gauss_sep_matches_gaussianblur(sol, scene):
    """Два прохода = GaussianBlur во float64: с заданным ksize и с размером по правилу ±3σ."""
    need(sol, "gauss_sep")
    got = np.asarray(sol["gauss_sep"](scene, 2.0, 13), np.float64)
    assert got.shape == scene.shape
    assert np.abs(got - cv2.GaussianBlur(np.float64(scene), (13, 13), 2.0)).max() < 1e-6
    auto = np.asarray(sol["gauss_sep"](scene, 1.5), np.float64)              # 2 · ceil(4.5) + 1 = 11
    assert np.abs(auto - cv2.GaussianBlur(np.float64(scene), (11, 11), 1.5)).max() < 1e-6


def test_gauss_sep_kernel_is_normalised(sol):
    """Ровный кадр остаётся ровным: сумма весов ядра равна единице."""
    need(sol, "gauss_sep")
    flat = np.full((30, 50), 100, np.uint8)
    got = np.asarray(sol["gauss_sep"](flat, 3.0), np.float64)
    assert got.shape == flat.shape and np.abs(got - 100).max() < 1e-9


def test_gauss_sep_equals_full_2d_kernel(sol, scene):
    """Сепарабельность: два прохода 1-D совпадают с одним проходом ядром K = v · vᵀ."""
    need(sol, "gauss_sep")
    v = cv2.getGaussianKernel(21, 3.5)
    full = filter2d(scene, v @ v.T)
    assert np.abs(np.asarray(sol["gauss_sep"](scene, 3.5, 21), np.float64) - full).max() < 1e-6


def test_table_shape_and_direct_psnr(sol, scene, noises):
    """db — (шумы, 1 + фильтры), столбец 0 — без фильтра; каждая клетка совпадает с прямым вызовом."""
    need(sol, "noise_filter_table")
    from cvcourse.metrics import psnr
    db, ms = sol["noise_filter_table"](scene, noises, FILTERS)
    db, ms = np.asarray(db, np.float64), np.asarray(ms, np.float64)
    assert db.shape == (2, 5) and ms.shape == (4,) and np.all(np.isfinite(db))
    for i, noisy in enumerate(noises.values()):
        assert abs(db[i, 0] - psnr(scene, noisy)) < 1e-6
        for j, f in enumerate(FILTERS.values()):
            assert abs(db[i, j + 1] - psnr(scene, f(noisy))) < 1e-6, "строка %d, столбец %d" % (i, j + 1)


def test_table_picks_the_right_filter(sol, scene, noises):
    """Под соль и перец — медиана; билатеральный на импульсном шуме хуже медианы; время — в миллисекундах."""
    need(sol, "noise_filter_table")
    db, ms = sol["noise_filter_table"](scene, noises, FILTERS)
    db, ms = np.asarray(db, np.float64), np.asarray(ms, np.float64)
    names = list(FILTERS)
    assert names[int(np.argmax(db[1, 1:]))] == "median 5", "на соли и перце лучший фильтр — медиана"
    assert db[1, 3] > db[1, 4] + 3, "билатеральный на соли и перце проигрывает медиане больше 3 дБ"
    assert np.all(db[:, 1:].max(1) > db[:, 0]), "лучший фильтр улучшает PSNR шумного кадра"
    assert np.all(ms > 0) and ms.max() < 5000, "время — в миллисекундах"
