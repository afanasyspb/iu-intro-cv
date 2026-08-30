# -*- coding: utf-8 -*-
"""Датасеты курса: скачивание с кэшем и проверкой контрольной суммы.

Данные **не хранятся в репозитории** — только реестр с адресами и SHA-256.
Кэш общий на машину (``~/.cache/cvcourse``), поэтому повторное занятие или
второй ноутбук не качают то же самое заново.

    from cvcourse import datasets
    path = datasets.fetch("middlebury-motorcycle")

Список того, что вообще есть: ``datasets.available()``.
"""
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

CACHE = os.environ.get("CVCOURSE_CACHE") or os.path.join(
    os.path.expanduser("~"), ".cache", "cvcourse")

#: Реестр. sha256=None означает «сумма ещё не зафиксирована» — такие записи
#: закрываются на неделе 0, до выдачи материалов студентам.
REGISTRY = {
    "middlebury-motorcycle": dict(
        url="https://vision.middlebury.edu/stereo/data/scenes2014/zip/Motorcycle-perfect.zip",
        sha256=None, kind="zip", used_in="L12, занятие 12, ДЗ 4",
        note="Стереопара с эталонной диспаратностью, метрика bad-2.0"),
    "sintel-clean-alley": dict(
        url="http://files.is.tue.mpg.de/sintel/MPI-Sintel-testing.zip",
        sha256=None, kind="zip", used_in="L13, занятие 13",
        note="Оптический поток с ground truth, метрика EPE"),
    "cdnet-highway": dict(
        url="http://jacarini.dinf.usherbrooke.ca/static/dataset/baseline/highway.zip",
        sha256=None, kind="zip", used_in="занятие 13",
        note="Вычитание фона, готовая разметка переднего плана"),
    "oxford-buildings-lite": dict(
        url=None, sha256=None, kind="dir", used_in="занятие 14",
        note="Урезанная коллекция для мини-поиска BoW; собирается вручную, см. data/README.md"),
    "oxford-affine-graf": dict(
        url="https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz",
        sha256=None, kind="tar", used_in="занятия 8 и 9",
        note="Шесть видов одной сцены с известными гомографиями — repeatability"),
    "kitti-odometry-00": dict(
        url=None, sha256=None, kind="dir", used_in="L15, обзорный блок по VO",
        note="Только для демонстрации на лекции; регистрация на сайте KITTI"),
}


def available():
    """Печатает реестр: что есть, где используется, скачано ли."""
    for name, e in sorted(REGISTRY.items()):
        p = os.path.join(CACHE, name)
        state = "скачан" if os.path.exists(p) else "нет"
        print("%-26s %-8s %-24s %s" % (name, state, e["used_in"], e["note"]))


def fetch(name, *, force=False, quiet=False):
    """Возвращает путь к каталогу датасета, скачав и распаковав при необходимости."""
    if name not in REGISTRY:
        raise KeyError("Неизвестный датасет %r. Доступные: %s"
                       % (name, ", ".join(sorted(REGISTRY))))
    e = REGISTRY[name]
    dst = os.path.join(CACHE, name)
    if os.path.exists(dst) and not force:
        return dst
    if not e["url"]:
        raise RuntimeError(
            "Датасет %r нельзя скачать автоматически: %s\n  Инструкция — в "
            "data/README.md соответствующего занятия." % (name, e["note"]))

    _mkdir(CACHE)
    tmp = dst + ".download"
    _download(e["url"], tmp, quiet=quiet)
    if e["sha256"]:
        got = sha256(tmp)
        if got != e["sha256"]:
            os.remove(tmp)
            raise RuntimeError(
                "Контрольная сумма не сошлась для %s\n  ожидалась %s\n  получена %s\n"
                "  Скачивание оборвалось или файл на сервере изменился."
                % (name, e["sha256"], got))
    elif not quiet:
        sys.stderr.write("  ! sha256 для %s ещё не зафиксирована\n" % name)

    _mkdir(dst)
    try:
        _extract(tmp, dst, e["kind"])
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dst


def sha256(path, *, chunk=1 << 20):
    """SHA-256 файла. Ей же фиксируются суммы в реестре."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def clear(name=None):
    """Удаляет кэш целиком или один датасет."""
    target = CACHE if name is None else os.path.join(CACHE, name)
    shutil.rmtree(target, ignore_errors=True)
    return target


# ------------------------------------------------------------ внутреннее ---
def _mkdir(p):
    if not os.path.isdir(p):
        os.makedirs(p)


def _download(url, dst, *, quiet=False):
    def hook(blocks, size, total):
        if quiet or total <= 0:
            return
        done = min(100, 100 * blocks * size // total)
        sys.stderr.write("\r  скачивание %d%% (%.1f МБ)" % (done, total / 1e6))
        sys.stderr.flush()
    try:
        urllib.request.urlretrieve(url, dst, hook)
    except Exception as ex:
        raise RuntimeError(
            "Не удалось скачать %s\n  %s: %s\n  Скачайте вручную и положите в %s"
            % (url, type(ex).__name__, ex, os.path.dirname(dst)))
    if not quiet:
        sys.stderr.write("\r  скачано: %s\n" % os.path.basename(dst))


def _extract(src, dst, kind):
    if kind == "zip":
        with zipfile.ZipFile(src) as z:
            z.extractall(dst)
    elif kind == "tar":
        with tarfile.open(src) as t:
            t.extractall(dst)          # noqa: S202 — источники в реестре доверенные
    else:
        shutil.copy(src, dst)
