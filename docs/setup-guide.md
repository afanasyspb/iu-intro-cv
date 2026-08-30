# Установка окружения

**Курс «Введение в компьютерное зрение» · Innopolis University · Fall 2026**

Сделайте это **до первого занятия**. На занятии время на установку не выделяется.

Проверка, что всё готово, — одна команда:

```bash
python tools/check_env.py
```

Ожидаемый вывод: `Окружение в порядке: OpenCV 4.14.0, cvcourse 1.0, Python 3.11`.

---

## Что ставим

| Компонент | Версия | Почему так |
|-----------|--------|-----------|
| Python | **3.11** (годится 3.10–3.12) | Не 3.8 — снят с поддержки в октябре 2024 |
| **opencv-contrib-python** | **`==4.14.0.94`** 🔒 | Точная версия, см. ниже |
| NumPy, Matplotlib, SciPy | из `requirements.txt` | — |
| scikit-image, scikit-learn | из `requirements.txt` | метрики, k-means и SVM |
| JupyterLab | ≥ 4.0 | или VS Code + расширение Jupyter |
| Git | любой | сдача через GitHub Classroom |
| **cvcourse** | из репозитория, `-e .` | библиотека курса; ставится той же командой `pip install -r requirements.txt` и после этого импортируется из любого каталога |

### Почему версия OpenCV закреплена точно

В OpenCV 5.0 **сломана обратная совместимость**: например, `cv2.HoughLinesP`
возвращает массив формы `(N, 4)` вместо `(N, 1, 4)`. Любой пример из туториала
или ответа на Stack Overflow написан под 4.x и на 5.0 упадёт — причём иногда
не с ошибкой, а с молча неверным результатом.

Кроме того, эталонные числа в ассертах ноутбуков и в автотестах привязаны к
конкретной сборке. Плавающая версия — это плавающие числа.

**Ставим `opencv-contrib-python`, а не `opencv-python`.** Без `contrib` не будет
`cv2.ximgproc` (WLS-фильтр, занятие 12) и `cv2.aruco` (ChArUco, занятие 11).
Если ставили `opencv-python` раньше — сначала удалите: два пакета в одном
окружении конфликтуют.

---

## Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip uninstall -y opencv-python opencv-python-headless
pip install -r requirements.txt
python tools/check_env.py
```

Если `Activate.ps1` не запускается — политика выполнения:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Кириллица в пути.** `cv2.imread` на Windows не открывает файлы, если в пути
есть кириллица, и молча возвращает `None`. В курсе для чтения используется
`cvcourse.io.imread` — он это обходит. Всё равно держите проект по пути без
русских букв и пробелов.

**Длинный путь.** Файлы JupyterLab лежат глубоко (`.venv\share\jupyter\labextensions\…`),
и если репозиторий клонирован в `C:\Users\Имя\Documents\Учёба\3 курс\…`, установка
падает с `OSError: [Errno 2] … HINT: Windows Long Path support`. Держите репозиторий
по короткому пути — `C:\cv\iu-intro-cv`.

---

## macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python tools/check_env.py
```

На Apple Silicon колёса `opencv-contrib-python` собираются под arm64 — ставится
без сборки из исходников. Если `pip` начал компилировать — значит, взялся
не тот Python; проверьте `python -c "import platform; print(platform.machine())"`,
должно быть `arm64`.

---

## Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python tools/check_env.py
```

Для показа окон `cv2.imshow` нужны системные библиотеки:

```bash
sudo apt install -y libgl1 libglib2.0-0
```

---

## Conda

```bash
conda env create -f environment.yml
conda activate iu-intro-cv
python tools/check_env.py
```

OpenCV ставится через `pip` внутри conda-окружения намеренно: в `conda-forge`
нет контриб-сборки нужной версии.

---

## Google Colab

Работает для большинства занятий, но **не для всех** — см. ограничения ниже.

```python
!pip install -q opencv-contrib-python==4.14.0.94
!pip install -q git+https://github.com/afanasyspb/iu-intro-cv.git
import cvcourse; print(cvcourse.__version__)
```

После установки OpenCV Colab попросит перезапустить среду выполнения — согласитесь.

### Чего в Colab нет

| Не работает | Где нужно | Что делать |
|-------------|-----------|-----------|
| `cv2.imshow` | везде | использовать `cvcourse.viz.show()` — работает в обеих средах |
| Трекбары, обработка мыши, `waitKey` | занятия 3, 7, 11 | локальный запуск |
| Потоковое видео с камеры | занятия 1, 3, 13, 14; ДЗ 2, ДЗ 3 | локальный запуск; в Colab доступен только одиночный снимок через JS |

**Занятия 1, 3, 13, 14 и ДЗ 2, ДЗ 3 требуют локального запуска.**
Для остального Colab — рабочий вариант.

---

## Запуск ноутбуков занятий

Все команды — из **корня репозитория**, с активированным окружением:

```bash
jupyter lab                       # откроется браузер; дальше labs/lab01-images/lab01.ipynb
```

В VS Code: открыть папку репозитория, в ноутбуке выбрать ядро (kernel) —
интерпретатор из `.venv`. Пакет `cvcourse` установлен в окружение, поэтому
ноутбук находит его из любого каталога; относительные пути внутри ноутбука
(`data/my_photo.jpg`) считаются от каталога занятия.

---

## Данные

Датасеты **не лежат в репозитории**. Качаются в кэш `~/.cache/cvcourse`
один раз на машину:

```python
from cvcourse import datasets
datasets.available()                       # что вообще есть
path = datasets.fetch("middlebury-motorcycle")
```

Путь кэша переопределяется переменной `CVCOURSE_CACHE`. Если университетская
сеть не пускает — скажите ассистенту **в начале занятия**, у него есть копия.

---

## Частые проблемы

| Симптом | Причина | Решение |
|---------|---------|---------|
| `ModuleNotFoundError: No module named 'cvcourse'` | библиотека курса не установлена в это окружение (или ядро ноутбука смотрит в другой Python) | из корня репозитория `pip install -r requirements.txt`; в Jupyter выбрать ядро из `.venv` |
| `OSError: [Errno 2] No such file or directory: '…\share\jupyter\labextensions\…'` при установке | слишком длинный путь к репозиторию (Windows, 260 знаков) | перенести репозиторий в `C:\cv\iu-intro-cv`, пересоздать `.venv` |
| `AttributeError: module 'cv2' has no attribute 'ximgproc'` | стоит `opencv-python` вместо контриб-сборки | `pip uninstall -y opencv-python` и переустановить по `requirements.txt` |
| `TypeError: 'NoneType' object is not subscriptable` после `imread` | файл не найден или кириллица в пути | использовать `cvcourse.io.imread` — он скажет, что именно не так |
| Лица на картинке синие | BGR показали как RGB | `cvcourse.viz.show()` либо `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| `200 + 100 = 44` | `uint8` переполняется молча | считать во `float32`, возвращаться через `np.clip(...).astype(np.uint8)` |
| Ядро висит после `cv2.imshow` | не вызван `waitKey` / не освобождён захват | `cvcourse.io.video_frames` освобождает камеру сам |
| Числа не совпали с эталонными | другая версия OpenCV | `python tools/check_env.py` |

---

## Быстрая проверка руками

```python
import cv2, numpy as np
from cvcourse import viz

img = np.zeros((200, 300, 3), np.uint8)
cv2.circle(img, (150, 100), 60, (0, 112, 192), -1)   # BGR: это синий круг
viz.show(img, "круг должен быть синим")
```

Круг оранжевый — значит, где-то перепутаны BGR и RGB.
