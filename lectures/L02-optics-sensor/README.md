# L2 — Формирование изображения: оптика и сенсор

| Что | Где |
|-----|-----|
| Слайды | `slides.pdf` — выкладывается накануне лекции |
| Живой код лекции | `demo/01_projection_optics.py` — демо-блок 1 (проекция, fx из FOV, дисторсия, ГРИП) · `demo/02_sensor_isp.py` — демо-блок 2 (мозаика Байера, ложный цвет, шум √N, rolling shutter) |
| Данные демо | `demo/iu-building.jpg` — фотография здания ИУ |
| Занятие с ассистентом | `labs/lab02-raw/` — «От RAW к RGB» |
| Чтение к следующей паре | Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., §2.3.2 (цвет) |

Запуск демо из корня репозитория (окружение — по `docs/setup-guide.md`):

```bash
python lectures/L02-optics-sensor/demo/01_projection_optics.py
python lectures/L02-optics-sensor/demo/02_sensor_isp.py
```

Скрипты печатают те же числа, что стоят на слайдах. Демо-блок 1 не требует
даже OpenCV — модель камеры это чистый NumPy.
