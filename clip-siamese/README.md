# CLIP-Siamese: Мультимодальное сопоставление товаров

# Setup
0. Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) with
```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
1. Install env `cross-encoder` with
```
micromamba create -f environment.yml
```
2. Use env `cross-encoder` in all notebooks in `cross-encoder/`

## Директории
- `data/images_*` - распакованные изображения для различных датасетов
- `data/tables_*` - файлы с табличными данными для различных датасетов
- `data/train_results/` - чекпоинты обученных моделей
- `data/train_results/tmp` - временные чекпоинты обученных моделей
TODO: вынести генерацию папки `tmp` из `data` в ноутбуке `contrastive_train_data-splitting-by-query.ipynb` чтобы не хранить временные чекпоинты вместе с основными
- `data/walkthough/` - видеогайды по ноутбукам в репозитории
- `reports` - отчеты

## Группы ноутбуков по назначению

0. **Remote data repo management**:
    1) `hf_data_download.ipynb` - для загрузки всех доступных данных из HF repo `INDEEPA/clip-siamese`;
    2) `hf_data_upload.ipynb` - для загрузки всех новых данных в HF repo `INDEEPA/clip-siamese`; 
    3) `hf_data_delete.ipynb` - удалить файлы из HF репо `INDEEPA/clip-siamese` с помощью регулярных выражений (удобнее чем вручную).

1. **Создание датасетов**: 
    1) `make_OZ_geo_5500.ipynb` - скачать изображения для товаров, объединить описания и экономические признаки из двух исходных файлов.
    2) `make_OZ_geo_5500_pairwise-regex.ipynb` - разметка регулярными выражениями

2. **Вычисление эмбеддингов**:
    1) `compute_embeddings_sbert.ipynb` - посчитать эмбеддинги с помощью Sentence Transformers
    2) `compute_embeddings_contrastive-siamese.ipynb` - посчитать эмбеддинги с помощью SiameseRuCLIP

3. **Обучение SiameseRuCLIP**: 
    1) `contrastive_train_data-spliting-by-query.ipynb` - обучить SiameseRuCLIP
    
4. **Тестирование и диагностика SiameseRuCLIP**
    1) `contrastive_test_embs_from-pairwise-rendered.ipynb` - получить предсказания модели SiameseRuCLIP на тестовом сете попарного набора целевых товаров с выбором порога по TPR@FPR на валидационном сете;
    2) `contrastive_test_embs_from-pairwise-queries.ipynb` - получить предсказания модели SiameseRuCLIP на попарном наборе целевых товаров с выбором порога по TPR@FPR;
    3) `contrastive_test_embs_from-source.ipynb` **(deprecated)** - получить top-k предсказаний модели SiameseRuCLIP на попарном наборе целевых товаров;
    4) `contrastive_test_embs_from-source_ruclip-only.ipynb` **(deprecated)** - получить top-k предсказаний модели **RuCLIP** на попарном наборе целевых товаров; 
    5) `contrastive_test.ipynb` **(deprecated)** - тестирование SiameseRuCLIP обученной с контрастным лоссом (логирование в W&B);
    6) `clip_siamese_test.ipynb` **(deprecated)** - тестирование SiameseRuCLIP обученной с лоссом CrossEntropy (логирование в W&B).