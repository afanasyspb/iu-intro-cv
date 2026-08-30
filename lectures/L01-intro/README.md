# L1 — Введение. Изображение и видео как данные

| Что | Где |
|-----|-----|
| Слайды | `slides.pdf` — выкладывается накануне лекции |
| Живой код лекции | `demo/01_image_as_array.py` — демо-блок 1 (массив, BGR, `uint8`, цикл против NumPy) · `demo/02_views_formats_video.py` — демо-блок 2 (срезы, `resize`, сжатие, видео) |
| Данные демо | `demo/iu-building.jpg` — фотография здания ИУ; синтетический ролик пишется во временный каталог при первом запуске |
| Занятие с ассистентом | `labs/lab01-images/` |
| Чтение к следующей паре | Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., гл. 1 |

Запуск демо из корня репозитория (окружение — по `docs/setup-guide.md`):

```bash
python lectures/L01-intro/demo/01_image_as_array.py
python lectures/L01-intro/demo/02_views_formats_video.py
```

Скрипты печатают те же числа, что стоят на слайдах. Тайминги и скорость
декодирования зависят от машины — на слайдах указан порядок величины.
