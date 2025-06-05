# OVERVIEW
В подпроекте `cross-encoder` планировалось обучать [Cross-Encoder](https://sbert.net/examples/cross_encoder/applications/README.html) чтобы повторить пайплайн [ecom-tech](https://habr.com/ru/companies/ecom_tech/articles/852646/).

В частности, была попытка разметить набор данных OZ_geo_5500 с помощью LLM - для каждого товара найти все дубликаты/похожие по содержанию (название + описание, без картинки).
С помощью LLM разметить набор данных не удалось; основные недостатки подхода - сложность составления точного промпта, нестабильность результатов запроса от вызова к вызову даже при низкой температуре и top-p.

# SETUP
0. Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) with
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
1. Install env `cross-encoder` with
```
micromamba create -f environment.yml
```
2. Use env `cross-encoder` in all notebooks in `cross-encoder/`

# REPORTS
    
## reports/LLM-single-stage.pdf
Анализ ошибок LLM для поиска конкурентов напрямую из полного набора товаров.

# NOTEBOOKS

## make_OZ_geo_5500_pairwise-LLM-single-stage.ipynb
Поиск конкурентов с помощью LLM напрямую из полного набора товаров.

## make_OZ_geo_5500_pairwise-LLM-attribute-extraction.ipynb
Поиск конкурентов с помощью LLM в два этапа:
- выделить атрибуты из описаний с помощью LLM
- найти матчи товаров по совпадающим атрибутам (точное совпадение по категориальным + совпадение по численным +-*заданный_зазор*% = матч)

## training_gooaq_bce.ipynb (training_gooaq_bce.py)
Пайплайн обучения кросс-энкодера из документации [Sentence Transformers](https://sbert.net/docs/sentence_transformer/training_overview.html)