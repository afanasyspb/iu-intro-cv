# -*- coding: utf-8 -*-
"""Метрики курса — посчитанные одинаково у всех.

Каждая функция возвращает **одно число** и имеет строку «что считается
хорошим значением»: метрика без представления о её шкале бесполезна.
Реализации намеренно короткие и читаемые — студент должен иметь возможность
открыть этот файл и увидеть формулу, а не чёрный ящик.
"""
import cv2
import numpy as np

__all__ = ["psnr", "ssim", "mse", "repeatability", "precision_recall",
           "epe", "bad_n", "iou", "ate_rmse", "reprojection_error"]


def _f32(a):
    a = np.asarray(a, np.float32)
    return a / 255.0 if a.max() > 1.5 else a


# ------------------------------------------------- качество изображения ----
def mse(a, b):
    """Средний квадрат ошибки в шкале [0, 1]."""
    x, y = _f32(a), _f32(b)
    if x.shape != y.shape:
        raise ValueError("Формы не совпадают: %s и %s" % (x.shape, y.shape))
    return float(np.mean((x - y) ** 2))


def psnr(a, b):
    """Пиковое отношение сигнал/шум, дБ. Выше — лучше.

    Ориентиры: **> 40 дБ** — разницу не видно, **30–40** — хорошо,
    **< 25** — артефакты заметны глазом. Бесконечность при полном совпадении.
    """
    e = mse(a, b)
    return float("inf") if e == 0 else float(10.0 * np.log10(1.0 / e))


def ssim(a, b, *, win=11, sigma=1.5):
    """Structural Similarity в [-1, 1]. **> 0.95** — отлично, **< 0.7** — плохо.

    В отличие от PSNR учитывает структуру, а не только попиксельную разницу:
    сдвиг яркости на всю картинку PSNR убивает, а SSIM почти нет.
    """
    x, y = _f32(a), _f32(b)
    if x.ndim == 3:
        return float(np.mean([ssim(x[..., i], y[..., i], win=win, sigma=sigma)
                              for i in range(x.shape[2])]))
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = (win, win)
    mx = cv2.GaussianBlur(x, k, sigma)
    my = cv2.GaussianBlur(y, k, sigma)
    mxx = cv2.GaussianBlur(x * x, k, sigma) - mx * mx
    myy = cv2.GaussianBlur(y * y, k, sigma) - my * my
    mxy = cv2.GaussianBlur(x * y, k, sigma) - mx * my
    s = ((2 * mx * my + C1) * (2 * mxy + C2)) / \
        ((mx ** 2 + my ** 2 + C1) * (mxx + myy + C2))

    # Рамку шириной в половину окна отбрасываем: там гауссово окно выходит за
    # край и статистики считаются по достроенным пикселям. Без этого значение
    # уезжает на десятую (0.332 против 0.369 на тестовой сцене) и расходится
    # со scikit-image; с обрезкой совпадает до шестого знака.
    pad = (win - 1) // 2
    if pad and s.shape[0] > 2 * pad and s.shape[1] > 2 * pad:
        s = s[pad:-pad, pad:-pad]
    return float(np.mean(s))


# ---------------------------------------------------------- детекторы -----
def repeatability(kp1, kp2, H, shape, *, thr=3.0):
    """Доля точек первого кадра, найденных и во втором. **> 0.6** — хороший детектор.

    ``H`` — известная гомография между кадрами (в курсе берётся из разметки
    Oxford Affine либо задаётся синтетически). ``thr`` — допуск в пикселях.
    """
    p1 = np.array([k.pt for k in kp1], np.float32).reshape(-1, 1, 2)
    p2 = np.array([k.pt for k in kp2], np.float32)
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    proj = cv2.perspectiveTransform(p1, np.asarray(H, np.float64)).reshape(-1, 2)
    h, w = shape[:2]
    inside = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
    if inside.sum() == 0:
        return 0.0
    d = np.linalg.norm(proj[inside][:, None, :] - p2[None, :, :], axis=2)
    return float((d.min(axis=1) <= thr).mean())


def precision_recall(matches, is_correct):
    """Precision и recall сопоставлений. ``is_correct`` — булев массив той же длины."""
    ok = np.asarray(is_correct, bool)
    n = len(matches)
    if n == 0:
        return 0.0, 0.0
    p = float(ok.sum()) / n
    r = float(ok.sum()) / max(1, int(ok.size))
    return p, r


# ------------------------------------------------------ оптический поток ---
def epe(flow, gt, *, mask=None):
    """End-point error: средняя длина вектора ошибки, пиксели. Ниже — лучше.

    Ориентиры на MPI Sintel clean: **< 2 px** — хорошо для классических методов,
    Farnebäck обычно даёт 4–8 px, Lucas–Kanade на разреженных точках — 1–3 px.
    """
    f, g = np.asarray(flow, np.float32), np.asarray(gt, np.float32)
    d = np.linalg.norm(f - g, axis=2)
    if mask is not None:
        d = d[np.asarray(mask, bool)]
    return float(np.mean(d)) if d.size else float("nan")


# -------------------------------------------------------------- стерео ----
def bad_n(disp, gt, *, n=2.0, mask=None):
    """Доля пикселей с ошибкой диспаратности больше ``n``. Стандарт Middlebury.

    ``bad-2.0`` — основная метрика курса. Ориентиры: StereoSGBM с WLS-фильтром
    на Middlebury 2014 half-size даёт **10–20 %**, StereoBM — 25–40 %.
    """
    d, g = np.asarray(disp, np.float32), np.asarray(gt, np.float32)
    valid = np.isfinite(g) & (g > 0)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    if valid.sum() == 0:
        return float("nan")
    return float((np.abs(d[valid] - g[valid]) > n).mean())


# ---------------------------------------------------- геометрия и трекинг --
def iou(box_a, box_b):
    """Intersection over Union двух прямоугольников ``(x, y, w, h)``.

    Трекинг считается успешным при **IoU > 0.5** на кадре.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter) / union if union > 0 else 0.0


def reprojection_error(obj_pts, img_pts, rvec, tvec, K, dist=None):
    """Средняя ошибка репроекции в пикселях — главный критерий калибровки.

    Ориентир для L11 и ДЗ 4: **< 0.5 px** — калибровка удалась, **> 1.0 px** —
    надо переснять набор кадров (чаще всего виноват включённый автофокус).
    """
    proj, _ = cv2.projectPoints(np.asarray(obj_pts, np.float32), rvec, tvec,
                                np.asarray(K, np.float64),
                                None if dist is None else np.asarray(dist, np.float64))
    d = np.asarray(img_pts, np.float32).reshape(-1, 2) - proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(d, axis=1)))


def ate_rmse(traj, gt):
    """Absolute Trajectory Error, RMSE по позициям. Для обзорного блока по VO (L15)."""
    a, b = np.asarray(traj, np.float64), np.asarray(gt, np.float64)
    if a.shape != b.shape:
        raise ValueError("Траектории разной длины: %s и %s" % (a.shape, b.shape))
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
