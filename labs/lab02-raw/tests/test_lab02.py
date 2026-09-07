# -*- coding: utf-8 -*-
"""Автотесты занятия 2 «От RAW к RGB». Самопроверка студента и проверка в CI.

Тесты работают против ноутбука: функции импортируются прямо из него, чтобы
не дублировать код и не дать ему разойтись. Какой ноутбук — по порядку:

  1. путь в переменной окружения ``CVCOURSE_LAB_NB`` (так CI ассистента гоняет решение);
  2. ``lab02_solution.ipynb`` рядом — появляется в репозитории после занятия;
  3. ``lab02.ipynb`` — заготовка, заполненная студентом; тесты незаполненных TODO
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
              os.path.join(HERE, "lab02_solution.ipynb"),
              os.path.join(HERE, "lab02.ipynb"))
NB = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
WANTED = ("bayer_masks", "demosaic_bilinear", "noise_curve")


@pytest.fixture(scope="module")
def sol():
    """Функции из ноутбука, выполненные в отдельном пространстве имён."""
    if NB is None:
        pytest.skip("нет ни lab02_solution.ipynb, ни lab02.ipynb")
    nb = json.load(io.open(NB, encoding="utf-8"))
    g = {"__name__": "lab02", "__todo__": set()}
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
        exec(compile(head, "<lab02>", "exec"), g)
    for name in WANTED:
        if name not in g:
            pytest.skip("в ноутбуке нет функции %s" % name)
    return g


def need(sol, *names):
    """Тест пропускается, если нужный TODO ещё не заполнен."""
    todo = [n for n in names if n in sol["__todo__"]]
    if todo:
        pytest.skip("не заполнен TODO: " + ", ".join(todo))


def to_mosaic(img):
    """Та же мозаика RGGB, что в опорном примере ноутбука."""
    m = np.zeros(img.shape[:2], np.uint8)
    m[0::2, 0::2] = img[0::2, 0::2, 2]
    m[0::2, 1::2] = img[0::2, 1::2, 1]
    m[1::2, 0::2] = img[1::2, 0::2, 1]
    m[1::2, 1::2] = img[1::2, 1::2, 0]
    return m


@pytest.fixture(scope="module")
def scene():
    """Цветная сцена с плавными градиентами и тонкими линиями, 240 × 320."""
    y, x = np.mgrid[0:240, 0:320]
    img = np.stack([(x * 0.6 + 40) % 256, (y * 0.8 + 20) % 256, ((x + y) * 0.4 + 60) % 256],
                   axis=-1).astype(np.uint8)
    cv2.rectangle(img, (60, 50), (200, 170), (30, 200, 90), -1)
    for k in range(220, 300, 6):
        cv2.line(img, (k, 30), (k, 210), (250, 250, 250), 1)
    return img


def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    e = (d * d).mean()
    return 99.0 if e == 0 else 10 * np.log10(255.0 ** 2 / e)


def test_environment():
    assert cv2.__version__.startswith("4.14"), (
        "Курс собран на OpenCV 4.14.0.94, найдено " + cv2.__version__)
    assert hasattr(cv2, "ximgproc"), "нужен opencv-contrib-python, а не opencv-python"


def test_masks_partition_rggb(sol):
    """Ровно один канал на пиксель, зелёных вдвое больше, R в (0,0), B в (1,1)."""
    need(sol, "bayer_masks")
    mR, mG, mB = sol["bayer_masks"]((240, 320))
    for m in (mR, mG, mB):
        assert m.shape == (240, 320) and m.dtype == bool
    total = mR.astype(int) + mG + mB
    assert total.min() == 1 and total.max() == 1
    assert mG.sum() == 2 * mR.sum() == 2 * mB.sum()
    assert mR[0, 0] and mG[0, 1] and mG[1, 0] and mB[1, 1]


def test_masks_follow_pattern_letters(sol):
    """Буквы pattern идут по строкам ячейки 2 × 2 — GRBG отличает строки от столбцов."""
    need(sol, "bayer_masks")
    mR, mG, mB = sol["bayer_masks"]((8, 8), "GRBG")
    assert mG[0, 0] and mR[0, 1] and mB[1, 0] and mG[1, 1]
    mR, mG, mB = sol["bayer_masks"]((8, 8), "BGGR")
    assert mB[0, 0] and mR[1, 1]


def test_masks_match_mosaic(sol, scene):
    """Под маской R в мозаике лежит красный канал фото."""
    need(sol, "bayer_masks")
    mosaic = to_mosaic(scene)
    mR, mG, mB = sol["bayer_masks"](mosaic.shape)
    assert np.array_equal(mosaic[mR], scene[..., 2][mR])
    assert np.array_equal(mosaic[mB], scene[..., 0][mB])


def test_demosaic_shape_dtype_and_measured_values(sol, scene):
    """Выход — BGR uint8 формы фото; измеренные значения не тронуты."""
    need(sol, "bayer_masks", "demosaic_bilinear")
    mosaic = to_mosaic(scene)
    masks = sol["bayer_masks"](mosaic.shape)
    out = sol["demosaic_bilinear"](mosaic, masks)
    assert out.shape == scene.shape and out.dtype == np.uint8
    mR, mG, mB = masks
    assert np.array_equal(out[..., 2][mR], mosaic[mR]), "измеренный R изменён"
    assert np.array_equal(out[..., 1][mG], mosaic[mG]), "измеренный G изменён"
    assert np.array_equal(out[..., 0][mB], mosaic[mB]), "измеренный B изменён"


def test_demosaic_equals_opencv_bilinear_inside(sol, scene):
    """Главный результат занятия: восемь строк дают то же, что cvtColor по умолчанию."""
    need(sol, "bayer_masks", "demosaic_bilinear")
    mosaic = to_mosaic(scene)
    out = sol["demosaic_bilinear"](mosaic, sol["bayer_masks"](mosaic.shape))
    ref = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2BGR)
    inner = (slice(2, -2), slice(2, -2))
    assert psnr(out[inner], ref[inner]) > 45, "внутри кадра результат должен совпадать с OpenCV"
    assert abs(psnr(out, scene) - psnr(ref, scene)) < 1.0


def test_demosaic_is_not_channel_swapped(sol, scene):
    """Перепутанные BGR/RGB — самая частая ошибка; ловится по PSNR к фото."""
    need(sol, "bayer_masks", "demosaic_bilinear")
    mosaic = to_mosaic(scene)
    out = sol["demosaic_bilinear"](mosaic, sol["bayer_masks"](mosaic.shape))
    assert psnr(out, scene) > psnr(out[..., ::-1], scene) + 3


def test_noise_curve_is_photon_transfer_line(sol):
    """Пуассоновский стек: дисперсия равна сигналу, наклон около 1, строки по возрастанию."""
    need(sol, "noise_curve")
    rng = np.random.default_rng(1)
    lam = np.linspace(20, 400, 320)[None, :].repeat(120, 0)
    stack = rng.poisson(lam, (24, 120, 320)).astype(np.float32)
    curve = sol["noise_curve"](stack, bins=8)
    curve = np.asarray(curve, np.float64)
    assert curve.ndim == 2 and curve.shape[1] == 2
    assert 4 <= len(curve) <= 8
    assert np.isfinite(curve).all()
    assert np.all(np.diff(curve[:, 0]) > 0)
    slope = np.polyfit(curve[:, 0], curve[:, 1], 1)[0]
    assert 0.8 < slope < 1.2, "дисперсия должна расти вместе с сигналом с наклоном ≈ 1"


def test_noise_curve_skips_empty_bins(sol):
    """Двухуровневая сцена: заполнены только две корзины, пустые пропущены."""
    need(sol, "noise_curve")
    rng = np.random.default_rng(2)
    lam = np.full((100, 200), 50.0)
    lam[:, 100:] = 300.0
    stack = rng.poisson(lam, (16, 100, 200)).astype(np.float32)
    curve = np.asarray(sol["noise_curve"](stack, bins=8), np.float64)
    assert len(curve) == 2
    assert abs(curve[0, 0] - 50) < 5 and abs(curve[1, 0] - 300) < 10
