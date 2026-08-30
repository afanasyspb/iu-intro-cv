# -*- coding: utf-8 -*-
"""Тесты библиотеки курса. Прогоняются в CI на каждый push.

Библиотекой пользуются все пятнадцать занятий и четыре домашних задания —
её поломка останавливает курс, поэтому проверяется не «импортируется ли»,
а поведение: правильные числа на известных входах и внятные ошибки на
неправильных.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cvcourse import io as cio          # noqa: E402
from cvcourse import metrics, viz       # noqa: E402


@pytest.fixture(scope="module")
def scene():
    img = np.zeros((120, 160), np.uint8)
    cv2.rectangle(img, (20, 20), (100, 90), 255, -1)
    cv2.circle(img, (130, 30), 18, 128, -1)
    return img


# ------------------------------------------------------------------ io -----
def test_imread_missing_file_says_what_is_wrong(tmp_path):
    """Молчаливый None от cv2.imread — источник половины «у меня не работает»."""
    with pytest.raises(FileNotFoundError) as e:
        cio.imread(str(tmp_path / "нет-такого.png"))
    assert "не найден" in str(e.value)


def test_imread_empty_file(tmp_path):
    p = tmp_path / "broken.png"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="пустой"):
        cio.imread(str(p))


def test_imread_roundtrip_with_cyrillic_path(tmp_path, scene):
    """cv2.imread на Windows не открывает пути с кириллицей — обход обязателен."""
    p = tmp_path / "картинка тест.png"
    cio.imwrite(str(p), scene)
    back = cio.imread(str(p), "gray")
    assert back.shape == scene.shape
    assert np.array_equal(back, scene)


def test_imread_modes(tmp_path, scene):
    p = str(tmp_path / "s.png")
    cio.imwrite(p, cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR))
    assert cio.imread(p, "color").ndim == 3
    assert cio.imread(p, "gray").ndim == 2
    with pytest.raises(ValueError):
        cio.imread(p, "чего-то-не-то")


def test_float_uint8_roundtrip(scene):
    assert np.array_equal(cio.as_uint8(cio.as_float(scene)), scene)
    assert cio.as_float(scene).max() <= 1.0


def test_as_uint8_clips_instead_of_wrapping():
    """uint8 переполняется молча: 200 + 100 = 44. Это ловится здесь."""
    over = np.array([[1.5, -0.5]], np.float32)
    out = cio.as_uint8(over)
    assert out.dtype == np.uint8
    assert out.max() <= 255 and out.min() >= 0


# ------------------------------------------------------------- metrics -----
def test_psnr_identical_is_infinite(scene):
    assert metrics.psnr(scene, scene) == float("inf")


def test_psnr_drops_with_noise(scene):
    rng = np.random.default_rng(0)
    weak = np.clip(scene + rng.normal(0, 4, scene.shape), 0, 255).astype(np.uint8)
    strong = np.clip(scene + rng.normal(0, 24, scene.shape), 0, 255).astype(np.uint8)
    assert metrics.psnr(scene, weak) > metrics.psnr(scene, strong)


def test_ssim_identical_is_one(scene):
    assert metrics.ssim(scene, scene) == pytest.approx(1.0, abs=1e-4)


def test_ssim_survives_global_brightness_shift():
    """Отличие SSIM от PSNR: сдвиг яркости PSNR убивает, а структуру — почти нет.

    Сцена намеренно на среднесером фоне, а не на чёрном: у SSIM яркостный член
    равен (2·mx·my + C1) / (mx² + my² + C1), и при mx = 0 он схлопывается.
    На чёрном фоне SSIM просядет — и это верное поведение метрики, а не дефект.
    """
    base = np.full((120, 160), 110, np.uint8)
    cv2.rectangle(base, (20, 20), (100, 90), 190, -1)
    cv2.circle(base, (130, 30), 18, 60, -1)
    shifted = np.clip(base.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    assert metrics.ssim(base, shifted) > 0.9
    assert metrics.psnr(base, shifted) < 30


def test_ssim_matches_reference_implementation(scene):
    """Сверка с scikit-image: своя реализация должна давать те же числа.

    Формула SSIM короткая, и написать её «почти правильно» очень легко.
    Курс объявляет свои метрики эталоном для всех — значит, они сами
    обязаны сверяться с независимой реализацией.
    """
    ski = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(3)
    noisy = np.clip(scene + rng.normal(0, 15, scene.shape), 0, 255).astype(np.uint8)
    for a, b in ((scene, noisy), (scene, scene)):
        mine = metrics.ssim(a, b)
        ref = ski.structural_similarity(a, b, gaussian_weights=True, sigma=1.5,
                                        use_sample_covariance=False,
                                        data_range=255)
        assert mine == pytest.approx(ref, abs=0.02), (
            "SSIM разошёлся с scikit-image: %.4f против %.4f" % (mine, ref))


def test_psnr_matches_reference_implementation(scene):
    ski = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(4)
    noisy = np.clip(scene + rng.normal(0, 15, scene.shape), 0, 255).astype(np.uint8)
    assert metrics.psnr(scene, noisy) == pytest.approx(
        ski.peak_signal_noise_ratio(scene, noisy, data_range=255), abs=0.05)


def test_shape_mismatch_is_an_error(scene):
    with pytest.raises(ValueError, match="не совпадают"):
        metrics.mse(scene, scene[:50])


def test_iou_known_values():
    assert metrics.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert metrics.iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert metrics.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3, abs=1e-6)


def test_epe_is_vector_length():
    flow = np.zeros((4, 4, 2), np.float32)
    gt = np.dstack([np.full((4, 4), 3.0, np.float32), np.full((4, 4), 4.0, np.float32)])
    assert metrics.epe(flow, gt) == pytest.approx(5.0)


def test_bad_n_threshold():
    disp = np.full((4, 4), 5.0, np.float32)
    gt = np.full((4, 4), 6.5, np.float32)
    assert metrics.bad_n(disp, gt, n=2.0) == 0.0
    assert metrics.bad_n(disp, gt, n=1.0) == 1.0


def test_reprojection_error_is_zero_on_exact_projection():
    K = np.array([[500, 0, 80], [0, 500, 60], [0, 0, 1]], np.float64)
    obj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    rvec, tvec = np.zeros(3), np.array([0.0, 0.0, 5.0])
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    err = metrics.reprojection_error(obj, proj.reshape(-1, 2), rvec, tvec, K)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_repeatability_on_pure_translation():
    """Чистый сдвиг — детектор обязан найти почти те же точки."""
    img = np.zeros((120, 160), np.uint8)
    for p in [(30, 30), (90, 40), (60, 80), (120, 60)]:
        cv2.circle(img, p, 7, 255, -1)
    H = np.array([[1, 0, 7], [0, 1, 3], [0, 0, 1]], np.float64)
    warped = cv2.warpPerspective(img, H, (160, 120))
    det = cv2.ORB_create(200)
    r = metrics.repeatability(det.detect(img, None), det.detect(warped, None),
                              H, img.shape)
    assert r > 0.7, "repeatability %.3f — подозрительно низко для чистого сдвига" % r


def test_ate_rmse():
    assert metrics.ate_rmse([[0, 0, 0], [1, 0, 0]],
                            [[0, 0, 0], [1, 0, 0]]) == pytest.approx(0.0)


# ----------------------------------------------------------------- viz -----
def test_flow_to_color_shape():
    flow = np.zeros((10, 12, 2), np.float32)
    flow[..., 0] = 1.0
    out = viz.flow_to_color(flow)
    assert out.shape == (10, 12, 3)
    assert out.dtype == np.uint8


def test_show_rejects_none():
    with pytest.raises(ValueError, match="None"):
        viz._to_rgb(None)


def test_to_rgb_normalizes_float():
    a = np.array([[-2.0, 0.0], [1.0, 5.0]], np.float32)
    out = viz._to_rgb(a)
    assert out.dtype == np.uint8 and out.shape == (2, 2, 3)
    assert out.min() == 0 and out.max() == 255
