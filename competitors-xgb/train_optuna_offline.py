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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score

import xgboost as xgb
import optuna 

MPSTATS_TOKEN = '660c308b02abc1.461169310352435175768397cb1f37c4df2a2561'
#'65ddd17d7a2011.184733406ca693eaa900f8cf86e212b476abc2cd'

class CompetitorsSearch:
    def __init__(self, sku_pairs = None, params_path:str='model_params_f1_skf_shuffle', is_debug:bool=True, save_latents=False) -> None: # TODO: change for test or prod
        # data for inference
        self.sku_pairs = sku_pairs
        self.is_need_pretrain = True if sku_pairs is None else False
        self.num_class = None
        self.cash = {'data':dict(), 'description':dict()}
        self.params_path = params_path
        self.xgboost_params_path = os.path.join(params_path, 'xgboost_model.json')
        self.scaler_params_path = os.path.join(params_path, 'std_scaler.bin')
        self.is_debug = is_debug
        self.df = pd.DataFrame()
        self.sbert = SentenceTransformer('all-distilroberta-v1')
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=self.device)
        self.clip = clip
        self.processor = processor
        self.images = None
        self.names = None
        self.images_urls = list()
        self.im_problems = None
        self.y = None
        self.save_latents = save_latents
        
    # def make_tg_report(self, text) -> None:
    #     token = '6498069099:AAFtdDZFR-A1h1F-8FvOpt6xIzqjCbdLdsc'
    #     method = 'sendMessage'
    #     chat_id = 324956476

    #     _ = requests.post(
    #             url='https://api.telegram.org/bot{0}/{1}'.format(token, method),
    #             data={'chat_id': chat_id, 'text': text}
    #         ).json()
        
    def get_shard_name(e) -> str:
        if 0 <= e <= 143:
            num = '01'
        elif 144 <= e <= 287:
            num = '02'
        elif 288 <= e <= 431:
            num = '03'
        elif 432 <= e <= 719:
            num = '04'
        elif 720 <= e <= 1007:
            num = '05'
        elif 1008 <= e <= 1061:
            num = '06'
        elif 1062 <= e <= 1115:
            num = '07'
        elif 1116 <= e <= 1169:
            num = '08'
        elif 1170 <= e <= 1313:
            num = '09'
        elif 1314 <= e <= 1601:
            num = '10'
        elif 1602 <= e <= 1655:
            num = '11'
        elif 1656 <= e <= 1919:
            num = '12'
        elif 1920 <= e <= 2045:
            num = '13'
        elif 2046 <= e <= 2189:
            num = '14'
        elif 2190 <= e <= 2405: 
            num = '15'
        elif 2406 <= e <= 2621:
            num = '16'
        else:
            num = '17'
        return f'//basket-{num}.wbbasket.ru/'

    def make_url_base(self, sku):
        vol_num = math.floor(sku / 100000)
        part_num = math.floor(sku / 1000)
        shard_name = self.get_shard_name(vol_num)
        result = f'https:{shard_name}vol{vol_num}/part{part_num}/{sku}'
        return result
    
    def make_desc_url(self, sku):
        return f'{self.make_url_base(sku)}/info/ru/card.json'
    
    def make_img_url(self, sku):
        return f'{self.make_url_base(sku)}/images/c246x328/1.jpg'
    
    def get_sku_description(self, sku):
        url = self.make_desc_url(sku)
        response = requests.get(url, timeout=(3, 30))
        # Check if request was successful (status code 200)
        if response.ok:
            return response.json()
        else:
            return None

    # def get_sku_image(self, sku):
    #     url = self.make_img_url(sku)
    #     img_data = requests.get(url).content
    #     try:
    #         img_data = Image.open(BytesIO(img_data))
    #         self.images_urls.append(url)
    #     except: 
    #         img_data = None
    #     return img_data

    def get_sku_image(self, url):
        img_data = requests.get(url).content
        try:
            img_data = Image.open(BytesIO(img_data))
        except: 
            img_data = None
        return img_data

    def get_sku_image_offline(sku, img_dataset_dir='data/images_7k'):
        """
        Load an image for a given SKU from the dataset path.
        It tries .jpg first then .webp.

        Parameters:
            sku (int or str): The SKU number.
            img_dataset_path (str): Directory path where images are stored.
        
        Returns:
            Image object if found and opened; otherwise, None.
        """
        for ext in ['.jpg', '.webp']:
            img_path = os.path.join(img_dataset_dir, f"{sku}{ext}")
            if os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                    image = Image.open(BytesIO(img_data))
                    # Ensure the image loads completely
                    image.load()
                    return image
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        return None

    def get_images_names(self, offline=True) -> Tuple[List[Image.Image], List[object]]:
        images, names, problems = list(), list(), list()
        for row in self.df.iterrows():
            row_num = row[0]
            row = row[1]

            if offline:
                img1 = self.get_sku_image_offline(int(row.sku_first))
                img2 = self.get_sku_image_offline(int(row.sku_second))
            else:
                img1 = self.get_sku_image(int(row.sku_first))
                img2 = self.get_sku_image(int(row.sku_second))

            name1, name2 = row.name_first, row.name_second
            if img1 is not None and img2 is not None:
                images.append(img1)
                images.append(img2)
                names.append(name1)
                names.append(name2)
            else:
                problems.append(row_num)

        images = images
        return images, names, problems

    def get_data(self, sku, date_from=None, date_to=None):
        """
        gets features of the product like balance, sales, rating, price, n_comments, discount, position
        Returns:
            dict or list of dicts (if timedelta calculated for more than one day)
        """
        # данные ЗА предыдущий день появляются в 03:00 AM, за текущий день никогда нет до завтра.
        # если будем просматривать промежуток в 00:00 - 03:00 AM, то будет уже и "завтра", но данных нет
        # будем всегда смотреть на позавчера
        if date_from is None:
            date_from = date.today() - timedelta(days=1)
        if date_to is None:
            date_to = date.today() - timedelta(days=1)
        payload = {}
        headers = {
                'x-Mpstats-TOKEN': MPSTATS_TOKEN,
                'Content-Type': 'application/json'
            }
        url = f"https://mpstats.io/api/wb/get/item/{sku}/sales?d1={date_from}&d2={date_to}&SKU={sku}"
        response = requests.request("GET", url, headers=headers, data=payload)
        resp = response.json() # zero because returned list with one object
        return resp

    def concat_options(self, options: List[dict]) -> str:
        s = ''
        if options is not None:
            for d in options:
                l = list(d.values())
                s += l[0] + ': ' + l[1] + '. '
        return s

    def get_df_row(self, sku:int) -> Tuple[dict, list, list]:
        # сначала посмотрим есть ли данные в кэше:
        if sku not in self.cash['data'].keys():
            try:
                data = self.get_data(sku)[0] # 0 because we take data for 1 day and get list with one object in result
            except:
                print(f'sku not found:{sku}')
                return None, None, None
            self.cash['data'][sku] = data
            sku_data = self.get_sku_description(sku)
            self.cash['description'][sku] = sku_data
        else:
            data = self.cash['data'][sku] # not using .get()
            sku_data = self.cash['description'][sku] 

        options = self.concat_options(sku_data.get('options'))
        data['description'] = sku_data.get('description', '')
        data['name'] = sku_data.get('imt_name', '')
        data['options'] = options
        data['sku'] = sku
        data['has_video'] = sku_data.get('media').get('has_video', 0)
        data['photo_count'] = sku_data.get('media').get('photo_count', 0)
        check_if_eq = [sku_data.get('vendor_code', ''), sku_data.get('nm_colors_names'), 
                    sku_data.get('selling', {}).get('brand_name', ''), 
                    sku_data.get('selling', {}).get('supplier_id', '')]
        return data, sku_data.get('colors'), check_if_eq # TODO: add colors default value
    
    def make_dataframe(self) -> pd.DataFrame:
        """
        загружает данные в датафрейм
        """
        first, second = [], []
        paired = []
        problems = []
        names = ['iseq_vendor', 'iseq_color', 'iseq_brand', 'iseq_supp']  
        for sku_first in self.sku_pairs.sku_first.unique():    # approx 2 min
            temp = self.sku_pairs[self.sku_pairs.sku_first == sku_first]
            data_first, relatives, eq1 = self.get_df_row(sku_first)
            if data_first is None:
                if self.is_debug:
                    print(sku_first)
                for row in temp.iterrows():
                    row = row[1]
                    problems.append([row.sku_first, row.sku_second])
                continue
            for sku_second in temp.sku_second:
                data_second, _, eq2 = self.get_df_row(sku_second)
                if data_second is None:
                    if self.is_debug:
                        print(sku_first, sku_second)
                    problems.append([sku_first, sku_second])
                    continue
                d = {names[i]:(1 if el1 == el2 else 0) for i, (el1, el2) in enumerate(zip(eq1, eq2))}
                if relatives is not None and sku_second in relatives:
                    d['are_related'] = 1
                else:
                    d['are_related'] = 0
                paired.append(d)
                second.append(data_second)
                first.append(data_first)
        if self.is_debug:    
            self.make_tg_report('Конец загрузки данных')
        # handling problems (~ timeouts by mpstats server)
        # Just iterating again
        for sku_first, sku_second in problems:
            data_first, relatives, eq1 = self.get_df_row(sku_first)
            data_second, _, eq2 = self.get_df_row(sku_second)
            if data_first is None or data_second is None:
                if self.is_debug:
                    print(sku_first, sku_second)
                continue
            d = {names[i]:(1 if el1 == el2 else 0) for i, (el1, el2) in enumerate(zip(eq1, eq2))}
            if relatives is not None and sku_second in relatives:
                d['are_related'] = 1
            else:
                d['are_related'] = 0
            paired.append(d)
            second.append(data_second)
            first.append(data_first)

        data_first = pd.DataFrame(first)
        data_second = pd.DataFrame(second)
        data_paired = pd.DataFrame(paired)

        cols_to_stay = ['balance', 'sales', 'rating', 'final_price', 'comments', \
                        'description', 'name', 'options', 'sku', 'has_video', 'photo_count'] 
        data_first = data_first[cols_to_stay]
        data_second = data_second[cols_to_stay]

        df = data_first.join(data_second, lsuffix="_first", rsuffix=("_second"))
        # additional position feature
        df = pd.concat([df, data_paired], axis=1)
        # handling NaNs
        df.fillna(0, inplace=True)
        self.df = df

    def get_scores(self):
        """
        Parameters:
            df: pd.DataFrame -- main source to take data from. 1346 rows. Faster then using requests
            sbert, -- SBERT instance
            clip, -- clip from RuCLIP 
            processor -- processor from RuCLIP
        """
        desc_first, opt_first = self.df.description_first, self.df.options_first
        desc_second, opt_second = self.df.description_second, self.df.options_second 

        emb_first = self.sbert.encode(desc_first, convert_to_tensor=True, show_progress_bar = False)
        emb_second = self.sbert.encode(desc_second, convert_to_tensor=True, show_progress_bar = False)
        desc_sim = np.diag(util.cos_sim(emb_first, emb_second).cpu().numpy())

        emb_first = self.sbert.encode(opt_first, convert_to_tensor=True, show_progress_bar = False)
        emb_second = self.sbert.encode(opt_second, convert_to_tensor=True, show_progress_bar = False)
        opt_sim = np.diag(util.cos_sim(emb_first, emb_second).cpu().numpy())
        
        classes = list(self.names)
        templates = ['{}', 'это {}', 'на картинке {}', 'товар {}']
        # predict
        predictor = ruclip.Predictor(self.clip, self.processor, self.device, bs=8, templates=templates)
        with torch.no_grad():
            text_latents = predictor.get_text_latents(classes)
            images_latents = predictor.get_image_latents(self.images)
        if self.save_latents:
            torch.save(images_latents, os.path.join(self.params_path, 'images_latents.pt')) # TODO: в след раз можно не сохранять
            torch.save(text_latents, os.path.join(self.params_path, 'text_latents.pt'))
        name_sim = []
        img_sim = []
        for ind in range(0, text_latents.shape[0], 2):
            first = text_latents[ind]
            second = text_latents[ind + 1]
            name_sim.append(util.cos_sim(first, second).cpu().numpy().squeeze())
            
            first = images_latents[ind]
            second = images_latents[ind + 1]
            img_sim.append(util.cos_sim(first, second).cpu().numpy().squeeze())
        return np.c_[desc_sim, opt_sim, name_sim, img_sim]
    
    def make_scoring(self) -> None:
        """
        Функция, находящая косинусные близости векторов описаний, options, названий товаров и изображений.
        Создаёт pd.DataFrame с добавленными 4 колонками 'desc_sim', 'opt_sim', 'name_sim', 'img_sim'.
        """
        scores = self.get_scores()
        scores_df = pd.DataFrame(scores, columns=['desc_sim', 'opt_sim', 'name_sim', 'img_sim'])
        self.df = pd.concat([self.df.copy(), scores_df], axis=1)

    def run(self,):
        if self.is_need_pretrain:
            raise 'Model is not pre-trained. Train it first then re-run it'
        xgboost_model = xgb.XGBClassifier()
        xgboost_model.load_model(self.xgboost_params_path) 
        scaler = joblib.load(self.scaler_params_path)

        self.make_dataframe() 
        self.make_scoring()
        df = self.df.drop(['sku_first', 'sku_second', 'name_first', 'description_first', # not changing df inside
                'name_second', 'description_second', 'options_first', 'options_second'], axis=1)
        
        features_scaled = scaler.transform(df)
        competitor_classes = xgboost_model.predict(features_scaled)
        if self.debug:
            self.make_tg_report('Competitors run ended working')
        return competitor_classes

    def train(self, data_path):
        df = pd.read_csv(data_path) #[:20]   # TODO: test
        df = df.sample(frac=1) # TODO: делаем перемешивание
        # remap = {0:0, 0.1:0, 0.5:0, 0.7:1, 0.9:1, 1:1} 
        # sku.replace({'y':remap}, inplace=True)
        # sku.label = sku.y.apply(int)
        self.y = df.label.copy()
        self.num_class = df.label.unique().shape[0]
        df.drop(columns='label', inplace=True)
        # sku.columns = ['sku_first', 'sku_second', 'y']
        # self.sku_pairs = sku[['sku_first', 'sku_second']]

        # self.make_dataframe()   
        self.df = df
        # self.get_images_names()
        # self.make_scoring()
        X = self.df.drop(['sku_first', 'sku_second', 'name_first', 'description_first', # not changing df inside
                'name_second', 'description_second', 'options_first', 'options_second',
                'image_url_first', 'image_url_second'], axis=1).copy()
        # полный датасет, будем на нём 
        y = self.y.copy()
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        optuna_params = self.optuna_gridsearch(X, y)
        clf = xgb.XGBClassifier(**optuna_params.params)
        clf.fit(X, y)

        if not os.path.isdir(self.params_path):
            os.mkdir(self.params_path)

        self.df = pd.concat([self.df, self.y.to_frame('label')], axis=1)
        self.df.to_csv(os.path.join(self.params_path, 'data.csv'), index=False)
        clf.save_model(self.xgboost_params_path)
        joblib.dump(scaler, self.scaler_params_path, compress=True)
        

    def optuna_gridsearch(self, X_train, y_train):
        def objective(param, train_x, train_y, valid_x, valid_y):
            dtrain = xgb.DMatrix(train_x, label=train_y)
            dvalid = xgb.DMatrix(valid_x, label=valid_y)
            
            bst = xgb.train(param, dtrain)
            preds = bst.predict(dvalid)
            pred_labels = np.rint(preds)
            accuracy = balanced_accuracy_score(valid_y, pred_labels)  # , average='weighted'
            # f1 = f1_score(valid_y, pred_labels)
            return accuracy

        def objective_cv(trial):
            param = {
                "verbosity": 0,
                "objective": "multi:softmax",
                "num_class" : self.num_class,
                # use exact for small dataset.
                "tree_method": "exact",
                # defines booster, gblinear for linear functions.
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }

            if param["booster"] in ["gbtree", "dart"]:
                # maximum depth of the tree, signifies complexity of the tree.
                param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
                # minimum child weight, larger the term more conservative the tree.
                param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
                # defines how selective algorithm is.
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

            skf = StratifiedKFold(n_splits=3) # вот здесь НЕ перемешиваем. возможно, стоит ещё перемешать сам датасет
            scores = []
            for train_index, test_index in skf.split(X_train, y_train):
                X_train_skf, X_test_skf = X_train[train_index], X_train[test_index]
                y_train_skf, y_test_skf = y_train.iloc[train_index], y_train.iloc[test_index]
                accuracy = objective(param, X_train_skf, y_train_skf, X_test_skf, y_test_skf)
                scores.append(accuracy)
            return np.mean(scores)


        study = optuna.create_study(direction="maximize")
        study.optimize(objective_cv, n_trials=300, timeout=600)
        return study.best_trial

if __name__ == '__main__':
    # train_path = 'labeled.csv'
    train_path = 'model_params_big_test/data.csv'
    competitors_search = CompetitorsSearch(save_latents=True)
    competitors_search.train(train_path)
    # competitors_search.make_tg_report('Переделались данные')