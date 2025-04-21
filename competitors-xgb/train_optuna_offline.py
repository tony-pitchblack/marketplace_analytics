import pandas as pd
import numpy as np

import requests
import os
import math
import joblib
from datetime import date, timedelta
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer, util
import ruclip
from PIL import Image
from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score

import xgboost as xgb
import optuna
from pathlib import Path

MPSTATS_TOKEN = '660c308b02abc1.461169310352435175768397cb1f37c4df2a2561'

class CompetitorsSearch:
    DROP_FEATURE_COLS = [
        'label', 'sku_first', 'sku_second',
        'name_first', 'description_first',
        'name_second', 'description_second',
        'options_first', 'options_second',
        'image_url_first', 'image_url_second',
        'category_id', 'category_name'
    ]

    def __init__(self, sku_pairs=None, params_path: str = 'model_params',
                 is_debug: bool = True, save_latents=False,
                 n_trials=300, timeout=600) -> None:
        self.sku_pairs = sku_pairs
        self.is_need_pretrain = sku_pairs is None
        self.num_class = None
        self.cache = {'data': {}, 'description': {}}
        Path(params_path).mkdir(parents=True, exist_ok=True)
        self.params_path = params_path
        self.xgboost_params_path = os.path.join(params_path, 'xgboost_model.json')
        self.scaler_params_path = os.path.join(params_path, 'std_scaler.bin')
        self.is_debug = is_debug
        self.df = pd.DataFrame()
        self.sbert = SentenceTransformer('all-distilroberta-v1')
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.clip, self.processor = ruclip.load('ruclip-vit-base-patch32-384', device=self.device)
        self.save_latents = save_latents
        self.n_trials = n_trials
        self.timeout = timeout

    @staticmethod
    def get_shard_name(e: int) -> str:
        ranges = [
            (143, '01'), (287, '02'), (431, '03'), (719, '04'),
            (1007, '05'), (1061, '06'), (1115, '07'), (1169, '08'),
            (1313, '09'), (1601, '10'), (1655, '11'), (1919, '12'),
            (2045, '13'), (2189, '14'), (2405, '15'), (2621, '16')
        ]
        num = next((n for max_e, n in ranges if e <= max_e), '17')
        return f'//basket-{num}.wbbasket.ru/'

    def make_url_base(self, sku: int) -> str:
        vol_num = sku // 100000
        part_num = sku // 1000
        return f'https:{self.get_shard_name(vol_num)}vol{vol_num}/part{part_num}/{sku}'

    def make_desc_url(self, sku: int) -> str:
        return f'{self.make_url_base(sku)}/info/ru/card.json'

    def get_sku_description(self, sku: int) -> dict:
        r = requests.get(self.make_desc_url(sku), timeout=(3, 30))
        return r.json() if r.ok else {}

    def _drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.DROP_FEATURE_COLS, errors='ignore')

    def train(self, data_path: str):
        df = pd.read_csv(data_path)
        df.to_csv(Path(self.params_path) / 'data.csv', index=False)
        assert {'category_id', 'category_name'}.issubset(df.columns)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.num_class = df['label'].nunique()

        strat = df[['label', 'category_id']].apply(lambda r: f"{r['label']}_{r['category_id']}" , axis=1)
        dev_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=strat)
        test_df.to_csv(Path(self.params_path) / 'data_test.csv', index=False)

        y_dev = dev_df['label']
        X_dev = self._drop_features(dev_df)
        cat_dev = dev_df['category_id']
        cat_name_dev = dev_df['category_name']

        best_trial, overall, per_cat = self.optuna_gridsearch(
            X_dev, y_dev, cat_dev, cat_name_dev
        )

        # Save overall splits & summary
        for i, m in enumerate(overall):
            pd.DataFrame([m]).to_csv(Path(self.params_path) / f'dev_results_split={i}.csv', index=False)
        pd.DataFrame(overall).mean().to_frame().T.to_csv(
            Path(self.params_path) / 'dev_results.csv', index=False
        )

        # Save per-category splits
        for i, df_cat in enumerate(per_cat):
            df_cat.to_csv(Path(self.params_path) / f'dev_results_per_category_split={i}.csv', index=False)

        # Aggregate per-category, ensure int
        (pd.concat(per_cat, ignore_index=True)
           .groupby(['category_id', 'category_name'], as_index=False)
           .agg({
               'category_size': 'mean',
               'accuracy': 'mean',
               'f1_score': 'mean',
               'precision': 'mean',
               'recall': 'mean'
           })
           .assign(category_size=lambda d: d['category_size'].astype(int))
           .to_csv(Path(self.params_path) / 'dev_results_per-category.csv', index=False)
        )

        # Fit best on dev
        scaler = StandardScaler()
        Xd = scaler.fit_transform(X_dev)
        clf = xgb.XGBClassifier(**best_trial.params)
        clf.fit(Xd, y_dev)
        clf.save_model(str(Path(self.params_path) / 'xgboost_model_fit_on_dev.json'))
        joblib.dump(scaler, Path(self.params_path) / 'std_scaler_fit_on_dev.bin', compress=True)

        # Evaluate on test
        Xt = self._drop_features(test_df)
        yp = clf.predict(scaler.transform(Xt))
        test_overall = {
            'accuracy': accuracy_score(test_df['label'], yp),
            'f1_score': f1_score(test_df['label'], yp, average='macro'),
            'precision': precision_score(test_df['label'], yp, average='macro', zero_division=0),
            'recall': recall_score(test_df['label'], yp, average='macro', zero_division=0)
        }
        pd.DataFrame([test_overall]).to_csv(Path(self.params_path) / 'test_results.csv', index=False)

        # Per-category test metrics
        (test_df.assign(y_pred=yp)
            .groupby(['category_id', 'category_name'], group_keys=False)
            .apply(
                lambda g: pd.Series({
                    'category_size': len(g),
                    'accuracy': accuracy_score(g['label'], g['y_pred']),
                    'f1_score': f1_score(g['label'], g['y_pred'], average='macro', zero_division=0),
                    'precision': precision_score(g['label'], g['y_pred'], average='macro', zero_division=0),
                    'recall': recall_score(g['label'], g['y_pred'], average='macro', zero_division=0)
                }), include_groups=False)
            .reset_index()
            .assign(category_size=lambda d: d['category_size'].astype(int))
            .to_csv(Path(self.params_path) / 'test_results_per-category.csv', index=False)
        )

        # Refit on all data
        all_df = pd.concat([dev_df, test_df], ignore_index=True)
        Xa = StandardScaler().fit_transform(self._drop_features(all_df))
        clf_all = xgb.XGBClassifier(**best_trial.params)
        clf_all.fit(Xa, all_df['label'])
        clf_all.save_model(str(Path(self.params_path) / 'xgboost_model.json'))
        joblib.dump(StandardScaler().fit(self._drop_features(all_df)), Path(self.params_path) / 'std_scaler.bin', compress=True)

    def optuna_gridsearch(
        self, X_dev: pd.DataFrame, y_dev: pd.Series,
        cat_dev: pd.Series, cat_name_dev: pd.Series
    ):
        def objective(trial):
            # Base params
            params = {
                'verbosity': 0,
                'objective': 'multi:softmax',
                'num_class': self.num_class,
                'tree_method': 'exact',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
            }
            # Only for tree-based boosters
            if params['booster'] in ['gbtree', 'dart']:
                params.update({
                    'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
                    'max_depth': trial.suggest_int('max_depth', 3, 9, step=2),
                    'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
                    'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                })
            # Dart-specific
            if params['booster'] == 'dart':
                params.update({
                    'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
                    'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
                    'rate_drop': trial.suggest_float('rate_drop', 1e-8, 1.0, log=True),
                    'skip_drop': trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)
                })

            skf = StratifiedKFold(n_splits=3)
            accs = []
            for tr, val in skf.split(X_dev, y_dev):
                Xtr = StandardScaler().fit_transform(X_dev.iloc[tr])
                clf = xgb.XGBClassifier(**params)
                clf.fit(Xtr, y_dev.iloc[tr])
                preds = clf.predict(StandardScaler().fit_transform(X_dev.iloc[val]))
                accs.append(balanced_accuracy_score(y_dev.iloc[val], preds))
            return np.mean(accs)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        best = study.best_trial.params

        overall, per_cat = [], []
        skf = StratifiedKFold(n_splits=3)
        for tr, val in skf.split(X_dev, y_dev):
            Xtr = StandardScaler().fit_transform(X_dev.iloc[tr])
            clf = xgb.XGBClassifier(**best)
            clf.fit(Xtr, y_dev.iloc[tr])
            ypred = clf.predict(StandardScaler().fit_transform(X_dev.iloc[val]))

            overall.append({
                'accuracy': accuracy_score(y_dev.iloc[val], ypred),
                'f1_score': f1_score(y_dev.iloc[val], ypred, average='macro', zero_division=0),
                'precision': precision_score(y_dev.iloc[val], ypred, average='macro', zero_division=0),
                'recall': recall_score(y_dev.iloc[val], ypred, average='macro', zero_division=0)
            })

            dfv = pd.DataFrame({
                'category_id':   cat_dev.iloc[val].values,
                'category_name': cat_name_dev.iloc[val].values,
                'y_true':        y_dev.iloc[val].values,
                'y_pred':        ypred
            })
            df_cat = (
                dfv.groupby(['category_id', 'category_name'], as_index=False)
                   .apply(
                       lambda g: pd.Series({
                           'category_size': len(g),
                           'accuracy': accuracy_score(g['y_true'], g['y_pred']),
                           'f1_score': f1_score(g['y_true'], g['y_pred'], average='macro', zero_division=0),
                           'precision': precision_score(g['y_true'], g['y_pred'], average='macro', zero_division=0),
                           'recall': recall_score(g['y_true'], g['y_pred'], average='macro', zero_division=0)
                       }),
                       include_groups=False
                   )
                   .reset_index(drop=True)
                   .astype({'category_size': 'int32'})
            )
            per_cat.append(df_cat)

        return study.best_trial, overall, per_cat

if __name__ == '__main__':
    # train_path = 'labeled.csv'
    # train_path = 'model_params_big_test/data.csv'

    # train_path = 'res_balanced_accuracy/data_clustered_regex_classes=2.csv'
    # params_path = 'stratified_clusters=2'

    # train_path = 'model_params_big_test/data_clustered_regex_classes=2.csv'
    # params_path = 'sims=True_stratified_clusters=2'

    train_path = 'res_balanced_accuracy/data_clustered_clusters=9_sku=211_model=gpt-4.1-mini.csv'
    params_path = 'stratified_clusters=9'

    # train_path = 'model_params_big_test/data_clustered_clusters=9_sku=211_model=gpt-4.1-mini.csv'
    # params_path = 'sims=True_stratified_clusters=9'

    cs = CompetitorsSearch(
        save_latents=True,
        params_path=params_path,
        n_trials=300,
        timeout=600
    )
    cs.train(train_path)