# L3 — Цвет, гистограммы и точечные операции

| Что | Где |
|-----|-----|
| Слайды | `slides.pdf` — выкладывается накануне лекции |
| Живой код лекции | `demo/01_color_histograms.py` — демо-блок 1 (цветовые пространства, LUT и гамма, гистограмма и эквализация, CLAHE) · `demo/02_threshold_morphology.py` — демо-блок 2 (четыре бинаризации, маска по цвету → морфология → bbox, связные компоненты, где ломается hue) |
| Данные демо | `demo/iu-building.jpg` — фотография здания ИУ · `demo/ball-day.jpg`, `demo/ball-lamp.jpg` — тот же кадр с мячом при двух освещениях · `demo/text-lamp.png`, `demo/text-flat.png`, `demo/text-gt.png` — страница текста при неравномерном и ровном свете и её эталон · `demo/mask-noisy.png` — маска с крапинками и дырами |
| Занятие с ассистентом | `labs/lab03-color/` — «Цвет, гистограммы, пороги, морфология» |
| Чтение к следующей паре | Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., §3.6 (геометрические преобразования) |

Запуск демо из корня репозитория (окружение — по `docs/setup-guide.md`):

```bash
python lectures/L03-color-histograms/demo/01_color_histograms.py
python lectures/L03-color-histograms/demo/02_threshold_morphology.py
```

Скрипты печатают те же числа, что стоят на слайдах. Тайминги демо 1.2
(`cv2.LUT` против `np.power`) зависят от машины — на слайде порядок величины.
