# Marketplace project - competitors detection (INDEEPA/Экспонента)
Для установки среды для каждого подпроекта есть свой `environment.yml` или `setup.py`.
См. README для каждого подпроекта отдельно.

## clip-siamese
Модель SiameseRuCLIP для идентификации похожих товаров по содержанию (название, изображение, описание), а также простая фильтрация кандидатов по экономическим признакам для определения кокурентов.
Получает 80-90+ точность на валидации/тесте на попарном наборе по целевым товарам (товарам продаца 'ИНТЕРТРЕЙД') из OZ_geo_5500. См. `clip-siamese/contrastive_test_embs_from-pairwise-rendered.ipynb`
При этом получает точность < 10 на всем наборе пар целевых товаров (в т.ч. которые попали в трейн). См. `clip-siamese/contrastive_test_embs_from-pairwise-queries.ipynb`

**Для большинства ноутбуков доступны скринкасты в `data/walthrhough`, которые можно скачать с помощью `hf_data_download.ipynb` из HF repo `INDEEPA/clip-siamese`.**

*TODO: исключить Data Leak в тренировочном скрипте или показать, что он несущественный*
*TODO: замерить метрики ранжирования*
*TODO: обучить на лоссе для ранжирования, сравнить с контрастной версией*
*TODO: сравнить SimameseRuCLIP обученный на лоссе для ранжирования с CrossEncoder (см. cross-encoder)*

## competitors-xgb
Модель для определения конкурентных товаров с помощью градиентного бустинга на экономических признаках + (опционально скоры похожести из RuCLIP/Sentence Transformers).

*TODO: Можно доработать чтобы принимать на вход скор похожести пары товаров из SiameseRuCLIP, а также добавить относительные признаки (см. competitors-xgb/reports/XGBoost_feature_importance_analysis.pdf)*

## cross-encoder
Планировалось обучать [Cross-Encoder](https://sbert.net/examples/cross_encoder/applications/README.html) чтобы повторить пайплайн [ecom-tech](https://habr.com/ru/companies/ecom_tech/articles/852646/).
В частности, была попытка разметить набор данных OZ_geo_5500 с помощью LLM - для каждого товара найти все дубликаты/похожие по содержанию (название + описание, без картинки).
С помощью LLM разметить набор данных не удалось.
Однако, удалось частично разметить OZ_geo_5500 с помощью регулярных выражений (см. `clip-siamese/make_OZ_geo_5500_pairwise-regex).

*TODO: обучить кросс-энкодер на разметке по регуляркам*

## ru-clip-cc12m
Легаси код для обучения ruclip.

*TODO: определить, он вообще лучше, чем исходные веса (от Сбера): [ruclip](https://github.com/ai-forever/ru-clip) в качестве backbone для SiameseRuCLIP?*