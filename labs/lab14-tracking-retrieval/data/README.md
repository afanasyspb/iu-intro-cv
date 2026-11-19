# Данные занятия 14: трекинг и поиск

Собственная процедурная синтетика курса, seed=14. Полностью офлайн: сеть, камера,
внешние датасеты и материалы другого репозитория не нужны. Источник — `../make.py`.
Плакаты и цветовой пример адаптированы из открытых демо L14 (02/07/08);
ролик двух целей расширен до 40 кадров относительно демо 04.

В Jupyter данные создаются в памяти `make_scene()`. Для экспорта из каталога лабы:

```powershell
python make.py --output data
```

Используйте Python окружения курса: `opencv-contrib-python==4.14.0.94`.
Скрипт создаёт `scene.npz` (без pickle) и `manifest.json` с формами, dtype и суммами.
Архив использует фиксированные ZIP timestamps. Его SHA-256 относится к байтам файла;
суммы массива ниже — SHA-256 сырых C-order байтов (shape/dtype указаны в manifest).
На иной сборке OpenCV извлечение SIFT может отличаться: не подменяйте контрольные суммы.

## Состав и границы измерений

- `color_frames/color_gt`: 40 кадров с растущей эллиптической целью, BGR uint8.
- `frames/boxes_gt/visible`: 40 кадров, два постоянных размера, пропуск t=7,8 у цели 0.
  GT — xywh физических целей. При пересечении HSV-компоненты сливаются; прогноз
  не считается измерением. GT не передаётся трекеру.
- `train_images/train_desc`: шесть отдельных плакатов и готовые SIFT-дескрипторы.
- `database`: семь изображений; 0 — положительный, 1–5 — отвлекающие, 6 — перестановка 0.
- `query`, `foreign`, `blank`: перспективный положительный, чужой текстурный, пустой запрос.
- `*_xy/*_desc`: координаты и SIFT float32 (N,128), согласованные по индексам.
- `query_H`: известное преобразование database[0] → query только для независимой проверки.
  `permutation` описывает перестановку 16 блоков, не используется при поиске.
- Обучение словаря — только на train_desc; IDF — только по базе. Запрос не обучает модель.

Это стенд для понимания ассоциации и геометрии поиска плоского экземпляра;
он не оценивает качество на естественном видео, Oxford Buildings или категориях объектов.
Данные воспроизводятся локально, сгенерированные бинарные файлы в Git не требуются.

## Измеренные SHA-256

| Файл | SHA-256 |
|---|---|
| manifest.json | `80574b5ca2ebd7420a66f172bcf4ea8ee8455173e99f59af62aef131c4065525` |
| scene.npz | `413da501f0d6e4a47099a08353126c0ce257c16250dad59a0eb30a10c5a499ab` |

| Массив | SHA-256 C-order |
|---|---|
| blank | `72045233810e6628dc59dd614b74bcc0fd7697532662c5e7f81519e03ca54a86` |
| blank_desc | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| blank_xy | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| boxes_gt | `d39b7cba38cdfec007a7cc75508cc1d0180edd603c43c82dcc001867ffa4cdaa` |
| color_frames | `6c9224257378acdfad8f98c38d23b40c3184ba8adea7cdc65e8da331131d2d5f` |
| color_gt | `7d791e75069f724faaf632cc0054f00488e1b38fd3a4b2188c00e76fdaf11093` |
| database | `3cddc338c53f6aac9fcea2aea6d09de7d7eea3b21f2146e87f03f4a0193c72d7` |
| db0_desc | `165f6209154c3ee3345a53d252474d13a13fb68b41d2a7d4bbe8ccb2384320f6` |
| db0_xy | `771e5447094376e71fe04f7a05e802437c25200e2e89aba65e132e0d610fb168` |
| db1_desc | `f26e87987074df49f3e1bf7253f49090aa1ae39828f322da9dfc258e618ec6f4` |
| db1_xy | `9c95a41c8e575ca924408f7748b1838cefd47ec4bf98e2f6e9d8527a8568150d` |
| db2_desc | `20b200555e524cb24bfc780aaf0528911f7746ae58fcfef032ecc0701d0cfa59` |
| db2_xy | `152e05d0b95e0d3f7af1c909f43ef21eff5e626592d6e6a69d3c8dda3a91bf2d` |
| db3_desc | `9934d2f1e5ef33749e8915557deca489385928e27ee6a4ba72da2c036da228fe` |
| db3_xy | `8d0181062a6dda8d4efac50fa49445a53b4e4e1397284925711883e78532cf75` |
| db4_desc | `e7f3bca6f7a69e5eef458a042d4bc9bb123fd28342329fa3c12aefe39c8afdb8` |
| db4_xy | `36fc03d46c5dfde4efa40186ac852fa306445b7014aadbfeb148cfcc6afc0328` |
| db5_desc | `caf3e3815dc169170ea7ba97ec02f293bb9b185f23a58765ab7ed13ea5669e23` |
| db5_xy | `8a00bd47c09afa137a1b7e403fa0429b55b9d7de49fac6c3e8e6bb17a51ba73a` |
| db6_desc | `75401057883faa378d98b5f64582df0651f6f4c27818ff5f2bea412b8ca1bc97` |
| db6_xy | `fe36892d0e6c0a02222ddc2362292a98a750582eaaac31d12c996f3fbeec55ee` |
| foreign | `da2c5a787a64b666ab8aacd68cd9830c259ed2b92ac0a60a5902450a1bf17f26` |
| foreign_desc | `cea46353c67c3ee767920d0f3176276c258f51a66dff3bf208fbe62a0269acb7` |
| foreign_xy | `87dab0df2caa53c6ebfbf6f0b0d88b5898461e0d392d5d0dab454d703eb29461` |
| frames | `4d58c77a5fe8a12d70740235ee256fd2b26b036ded5a19853249977fc07d74d8` |
| permutation | `0f7330028917aa93819ab44ef2e3717261c5e3f421c75670064ccb526faa0126` |
| query | `5ace99136526285eef133ffd0fd5a22bd77cc838fc711d4b933d2d2e43fbfd3f` |
| query_H | `df1dcea9b72d6fd7fba408acf36b0082d9d5e04955fba18a7b54d943927369a1` |
| query_desc | `27464a66b0321e8281ad506c7da19ed5e7c5e58d614797982be31a2a01a69267` |
| query_xy | `4e4737be2f106bf793f400aeb240ea0f9da8c60918bbf67907bfbdf175cdb337` |
| train_desc | `581531f80f4a288247d7b17defdd70a93ee3238908a6809f0ab8beff910093ca` |
| train_images | `b6120aff52e892bb4b09970b9b8708f5aeae80f6defb91c36fd2aad005784b31` |
| visible | `41ccfb73301a888872bd1a7846323ef9c75ee029580f0d32317536ab28864241` |
