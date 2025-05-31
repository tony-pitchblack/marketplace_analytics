# SETUP
Run `./setup.sh` to create micromamba env `competitors-xgb`.
Use this env for all notebooks & scripts in competitors-xgb/

**NOTE: При установке ruclip есть конфликт:**
```
ruclip requires huggingface_hub==0.2.1
sentence_transformers requires huggingface_hub≥0.30.1
```
Так как ruclip давно не обновлялся и использует древний hugginface_hub, то помог форк `https://github.com/tony-pitchblack/ru-clip.git` с исправлением `ru-clip/ruclip
/__init__.py` (включен в requirements.txt вместо `ruclip`)

# NOTEBOOKS / FOLDERS

## make_pairwise_table_OZ_geo_5500_all_query_pairs.ipynb
Создает попарный набор данных для XGBoost.
Создает всевозможные пары <`sku_query`, `sku_candidate`>, где `sku_candidate` - все остальные (не целевые товары) из исходного набора товаров.
NOTE: `sku_query` = товары для целевого продавца, например "ИНТЕРТРЕЙД"

## make_pairwise_table_OZ_geo_5500_top_k_query_ruclip.ipynb
Создает попарный набор данных для XGBoost.
Выбирает top-k кандидатов для каждого `sku_query` с помощью ruclip, обученного Сбером.

## train_optuna_offline.py
Обучение XGBoost.
Последняя версия делит на `dev` / `test`.
Подбирает гиперпараметры кроссвалидацией на `dev`, обучает лучшую модель на всем `dev` и тестирует на `test`.
Раньше модель обучалась на всех данных.

В __main__() указать:
- `train_path` - путь к файлу с данными для обучения и валидации
- `params_path` - путь для сохранения весов и сплитов данных

Использует все колонки в качестве признаков, кроме записанных в исключения:
```
    DROP_FEATURE_COLS = [
        'label', 'sku_first', 'sku_second',
        'name_first', 'description_first',
        'name_second', 'description_second',
        'options_first', 'options_second',
        'image_url_first', 'image_url_second',
        'category_id', 'category_name'
    ]
```

*TODO: сделать нормальный скрипт с аргументами*

## debug
Содержит попарный датасет `Wildberries-5k-paired` - он же `labeled.csv`, а также ручную разметку `sku_labeled_original_elena.csv` и файлы, в разных формах повторяющие train_optuna.py для дебага (единственное  показывает как по )
- `debug.ipynb` подгружает картинки для `labeled.csv` в онлайне из MPSTATS (необходим валидный ключ).
- `debug_offline.inpynv` использует картинки для `labeled.csv` из датасета картинок `images_7k` по `sku`.

## xgb_inference.ipynb
Инференс XGBoost с выбором чекпоинта.
Опциональное отображение результатов `dev_results_*` / `test_results_*` (если сохранены с помощью последней версии `train_optuna_offine.py`)
Анализ важности признаков с помощью SHAP.
Вывод топ-k конкурентов в пагадигме где XGBoost - финальная модель.

## Чекпоинты XGBoost

Каждая из папок ниже содержит чекпоинты модели XGBoost, скейлера, данные для обучения.
Опционально содержит тестовую выборку и результаты `dev_results_*` / `test_results_*` (с помощью последней версии `train_optuna_offine.py`):
- model_params_big_test
- res_balanced_accuracy
- res_f1
- stratified_clusters=9 # есть группировка товаров по категориям (кластерам)
- sims=True_stratified_clusters=9 # включает признаки `name_sim`, `img_sim`, `desc_sim`, `opt_sim` => sims=True

## make_category_clustering.ipynb
Выделение категорий товаров с помощью LLM для возможности стратифицированного деления на train/val/test (равномерно по категориям).
Использует попарные файлы `data.csv` люого из чекпоинтов XGBoost (см. выше) чтобы по полям `name_first`, `name_second` выделить категории товаров.

## analyze_sentence_embeddings.ipynb
Языковые модели не понимают числа (по крайней мере не LLM).
Максимальная ранговая корреляция можеду попарными расстояниями эмбеддингов и евклидовыми расстояниями 0.8, что означает использовать извлеченные числовые признаки напрямую более эффективно чем полагаться на то, что текстовая модель различит размеры товаров.

## analyze_balance_sales_distributions.ipynb
Анализ распределения призаков `sales` и `balance` по классам в файле data/tables_OZ_geo_5500/source/конкуренты_АП.xlsx.
Это попытка понять, как делают предсказания специалисты (неудачная) - в файле data/tables_OZ_geo_5500/source/конкуренты_АП.xlsx зеленым отмечены хорошие предсказания модели `model_params_big_test`, которая в т.ч. похожести товаров `name_sim`, `img_sim`, `desc_sim`, `opt_sim` и несмотря на то, что на Wildberries-5k-paired (labeled) дает точность > 75%, точность ранжирования на картах OZ_geo_5500 составляет всего 15% (согласно оценке экспертов в файле data/tables_OZ_geo_5500/source/конкуренты_АП.xlsx).
Несмотря на то, что сложно понять, как принимают решения специалисты, интуитивно кажется, что фильтрация по экономическим признакам хорошо формализуема и табличная модель не сильно улучшит ситуацию.