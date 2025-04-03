import pandas as pd

import requests
import os

import joblib
import xgboost as xgb
from datetime import date, timedelta
import numpy as np

import ruclip
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple
from PIL import Image
from io import BytesIO
import math

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

import optuna 

MPSTATS_TOKEN = '65ddd17d7a2011.184733406ca693eaa900f8cf86e212b476abc2cd'
cash = {'data':dict(), 'description':dict()}
sbert = SentenceTransformer('all-distilroberta-v1')
clip, processor = ruclip.load('ruclip-vit-base-patch32-384')
device = 'cpu'
data_path = 'data/sku_labeled_original_elena.csv'
is_debug = True

def make_tg_report(text) -> None:
    token = '6498069099:AAFtdDZFR-A1h1F-8FvOpt6xIzqjCbdLdsc'
    method = 'sendMessage'
    chat_id = 324956476
    _ = requests.post(
            url='https://api.telegram.org/bot{0}/{1}'.format(token, method),
            data={'chat_id': chat_id, 'text': text}
        ).json()
    
def get_shard_name(e):
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
    else:
        num = '13'
    return f'//basket-{num}.wb.ru/'

def make_url_base(sku):
    vol_num = math.floor(sku / 100000)
    part_num = math.floor(sku / 1000)
    shard_name = get_shard_name(vol_num)
    result = f'https:{shard_name}vol{vol_num}/part{part_num}/{sku}'
    return result

def make_desc_url(sku):
    return f'{make_url_base(sku)}/info/ru/card.json'

def make_img_url(sku):
    return f'{make_url_base(sku)}/images/c246x328/1.jpg'

def get_sku_description(sku):
    url = make_desc_url(sku)
    response = requests.get(url, timeout=(3, 30))
    # Check if request was successful (status code 200)
    if response.ok:
        return response.json()
    else:
        return None

def get_sku_image(sku):
    url = make_img_url(sku)
    img_data = requests.get(url).content
    try:
        img_data = Image.open(BytesIO(img_data))
        # images_urls.append(url)
    except: 
        img_data = None
    return img_data

def get_images_names() -> Tuple[List[Image.Image], List[object]]:
    images, names, problems = list(), list(), list()
    for row in df.iterrows():
        row_num = row[0]
        row = row[1]
        img1 = get_sku_image(int(row.sku_first))
        img2 = get_sku_image(int(row.sku_second))
        name1, name2 = row.name_first, row.name_second
        if img1 is not None and img2 is not None:
            images.append(img1)
            images.append(img2)
            names.append(name1)
            names.append(name2)
        else:
            problems.append(row_num)
    images = images
    # im_problems = problems
    return images, names, problems

def get_data(sku, date_from=None, date_to=None):
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

def concat_options(options: List[dict]) -> str:
    s = ''
    if options is not None:
        for d in options:
            l = list(d.values())
            s += l[0] + ': ' + l[1] + '. '
    return s

def get_df_row(sku:int) -> Tuple[dict, list, list]:
    # сначала посмотрим есть ли данные в кэше:
    if sku not in cash['data'].keys():
        try:
            data = get_data(sku)[0] # 0 because we take data for 1 day and get list with one object in result
        except:
            print(f'sku not found:{sku}')
            return None, None, None
        # print(sku)
        cash['data'][sku] = data
        sku_data = get_sku_description(sku)
        cash['description'][sku] = sku_data
    else:
        data = cash['data'][sku] # not using .get()
        sku_data = cash['description'][sku] 
    options = concat_options(sku_data.get('options'))
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



sku = pd.read_csv(data_path)[:20]   # TODO: test
remap = {0:0, 0.1:0, 0.5:0, 0.7:1, 0.9:1, 1:1} 
sku.replace({'y':remap}, inplace=True)
sku.y = sku.y.apply(int)
y = sku.y.copy()
num_class = sku.y.unique().shape[0]
sku.columns = ['sku_first', 'sku_second', 'y']
sku_pairs = sku[['sku_first', 'sku_second']]


first, second = [], []
paired = []
problems = []
names = ['iseq_vendor', 'iseq_color', 'iseq_brand', 'iseq_supp']  
for sku_first in sku_pairs.sku_first.unique():    # approx 2 min
    temp = sku_pairs[sku_pairs.sku_first == sku_first]
    data_first, relatives, eq1 = get_df_row(sku_first)
    if data_first is None:
        if is_debug:
            print(sku_first)
        for row in temp.iterrows():
            row = row[1]
            problems.append([row.sku_first, row.sku_second])
        continue
    for sku_second in temp.sku_second:
        data_second, _, eq2 = get_df_row(sku_second)
        if data_second is None:
            if is_debug:
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
if is_debug:    
    make_tg_report('Конец загрузки данных')
# handling problems (~ timeouts by mpstats server)
# Just iterating again
for sku_first, sku_second in problems:
    data_first, relatives, eq1 = get_df_row(sku_first)
    data_second, _, eq2 = get_df_row(sku_second)
    if data_first is None or data_second is None:
        if is_debug:
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
df = df

def get_scores():
    """
    Parameters:
        df: pd.DataFrame -- main source to take data from. 1346 rows. Faster then using requests
        sbert, -- SBERT instance
        clip, -- clip from RuCLIP 
        processor -- processor from RuCLIP
    """
    desc_first, opt_first = df.description_first, df.options_first
    desc_second, opt_second = df.description_second, df.options_second 
    emb_first = sbert.encode(desc_first, convert_to_tensor=True, show_progress_bar = False)
    emb_second = sbert.encode(desc_second, convert_to_tensor=True, show_progress_bar = False)
    desc_sim = np.diag(util.cos_sim(emb_first, emb_second).cpu().numpy())
    emb_first = sbert.encode(opt_first, convert_to_tensor=True, show_progress_bar = False)
    emb_second = sbert.encode(opt_second, convert_to_tensor=True, show_progress_bar = False)
    opt_sim = np.diag(util.cos_sim(emb_first, emb_second).cpu().numpy())
    images, names, problems_ids = get_images_names()
    id_to_del = ~df.index.isin(problems_ids)
    df = df[id_to_del]
    # print(df.shape)
    y = y[id_to_del]
    # print(y.shape)q
    if is_debug:
        print(f'ids of images that did not open: {problems_ids}')
    desc_sim = np.delete(desc_sim, problems_ids)
    opt_sim = np.delete(opt_sim, problems_ids)
    # print(desc_sim.shape)
    # print(opt_sim.shape)
    classes = list(names)
    # print(len(classes))
    templates = ['{}', 'это {}', 'на картинке {}', 'товар {}']
    # predict
    predictor = ruclip.Predictor(clip, processor, device, bs=8, templates=templates)
    with torch.no_grad():
        text_latents = predictor.get_text_latents(classes)
        images_latents = predictor.get_image_latents(images)
    name_sim = []
    img_sim = []
    # print(text_latents.shape)
    for ind in range(0, text_latents.shape[0], 2):
        first = text_latents[ind]
        second = text_latents[ind + 1]
        name_sim.append(util.cos_sim(first, second).cpu().numpy().squeeze())
        first = images_latents[ind]
        second = images_latents[ind + 1]
        img_sim.append(util.cos_sim(first, second).cpu().numpy().squeeze())
    # print(len(name_sim))
    # print(len(img_sim))
    return np.c_[desc_sim, opt_sim, name_sim, img_sim]

scores = get_scores()
# print(scores.shape)
scores_df = pd.DataFrame(scores, columns=['desc_sim', 'opt_sim', 'name_sim', 'img_sim'])
# print(scores_df.shape)
df = pd.concat([df.copy(), scores_df], axis=1)