import pandas as pd
import requests
from datetime import date, timedelta
import yaml
import os

with open(os.path.join('source', 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)
MPSTATS_TOKEN = config['mpstats_token']
    
def get_sells(userid, sku, date_from = '2023-01-01', date_to = '2023-10-01', skip = 0, take_num = 1000):
    url = 'https://api.indeepa.com/prod/analytics2/getreportdatabyid'
    payload = {
        'reportid': '7edad54d-713c-44a9-a863-73dfb1706da3', 
        'userid': userid, 
        'token' : '3a4c176f-f0e9-4169-8292-173ec4e95db7', 
        'skip' : skip,
        'take' : take_num,
        'parameters.vardateFrom' : date_from,
        'parameters.vardateTo' : date_to,
        'parameters.varNmId' : sku
    }
    return requests.get(url, params=payload, timeout=(3, 30))


def get_all_sells(userid, sku, date_from, date_to):
    sells = pd.DataFrame()
    go_on = True
    skip = 0
    while go_on:
        try:
            lst = get_sells(userid, sku, date_from, date_to, skip=skip).json()
            lst = pd.DataFrame(lst)
            sells = pd.concat([sells, lst])
            if lst.shape[0] != 1000:
                go_on = False
            skip += 1000
        except:
            print('Some error occured')
            go_on = False

    sells.drop_duplicates(inplace=True)
    sells['DateUtc'] = pd.to_datetime(sells['DateUtc']).dt.date
    sells[['SaleQty', 'OrderQty']] = \
        sells[['SaleQty', 'OrderQty']].astype(float)
    sells.NmId = sells.NmId.astype(int)
    sells = sells[sells.SaleQty >= 0]
    sells.set_index(['NmId', 'DateUtc'], inplace = True)
    sells.drop(['WarehouseName', 'MpSku', 'SaleAmount', 'OrderAmount'], 
               inplace=True, axis=1) 
    # складываем одинаковые дни
    sells = sells.groupby(level=[0, 1]).sum()
    group_index = 'NmId' 
    sells = sells.reset_index(group_index).groupby(group_index) \
            .apply(lambda x: x.asfreq('D')).drop(group_index, axis=1).fillna(0)
    sells.sort_index(inplace=True)
    sells.reset_index(inplace=True)
    sells.rename(columns={'DateUtc':'date', 'NmId':'sku'}, inplace=True)
    return sells

def make_mpstats_request(sku, date_from=date.today() - timedelta(days=1), date_to=date.today()):
    """
    gets features of the product like balance, sales, rating, price, n_comments, discount, position
    Returns:
        dict or list of dicts (if timedelta calculated for more than one day)
    """
    headers = {
            'x-Mpstats-TOKEN': MPSTATS_TOKEN,
            'Content-Type': 'application/json'
        }
    url = f"https://mpstats.io/api/wb/get/item/{sku}/sales?d1={date_from}&d2={date_to}&SKU={sku}&fbs=0"
    try:
        response = requests.request("GET", url, headers=headers, timeout=(3, 30))
        return response
    except:
        print('Timed out')

def get_mpstats_year(sku, date_from=date.today()-timedelta(days = 184), date_to=date.today(), halfyears=2):
    """
    Parameters:
        - sku:int, NmId товара
        - halfyears:int = 2, количество полугодий, за которые собирать статистику. 

    Функция, отдающая данные за год для указанного SKU.
    Чтобы получить список данных, наберите `get_data(<sku>)`
    По умолчанию отдаётся датафрейм со следующими столбцами:

    `'no_data', 'data', 'balance', 'sales', 'rating', 'price', 'final_price', `
    `'is_new', 'comments', 'discount', 'basic_sale', 'basic_price',`
    `'promo_sale', 'client_sale', 'client_price', 'categories_cnt',`
    `'visibility', 'position', 'promosale', 'clientsale', 'clientprice', 'sku'`
    """
    temp = []
    for _ in range(halfyears): # берём за год
        date_from = date_to - timedelta(days = 184)
        resp = make_mpstats_request(sku, date_from, date_to)
        if resp is None:
            break # Timed Out
        elif resp.ok:
            resp = resp.json()
        else:
            print(resp.status_code)
            print(resp.content)
            break
        temp.extend(resp)
        date_to = date_from
        if resp[0]['no_data'] != 0: # на новом запросе у нас данные, для которых нет записей
            break
    df = pd.DataFrame(temp)
    df['sku'] = sku
    df = df[df.no_data == 0].reset_index(drop=True)
    df = df.drop(columns=['price']).rename(columns={'data':'date', 'final_price':'price'})
    df = df[::-1].reset_index(drop=True) # переворачиваем 
    return df[['date', 'sales', 'price', 'balance']]

def get_data(sku, userid='0517ba04-3f9d-4817-8d2c-96d1aacac050', 
             date_from = date.today() - timedelta(days=366), 
             date_to = date.today(),
             verbose=False):
    """
    Чтобы получить список данных из стандартного кабинета для одного года, наберите `get_data(<sku>)`
    Данная функция использует источник mpstats для забора данных цен, а источник indeepa для забора данных о
        покупках и заказах.
    """
    try:
        sells = get_all_sells(userid, sku, date_from, date_to)
        if verbose:
            print('Взяли данные по заказам у Индипы')
    except KeyError:
        if verbose:
            print(f'И Индипы нет данных по заказам для этого sku ({sku}), используем полностью источник mpstats')
        res = get_mpstats_year(sku, date_from, date_to)
        res['sku'] = sku
        return res     
    try:
        setts = get_mpstats_year(sku, date_from, date_to)
        if verbose:
            print('Взяли данные по ценам в mpstats')
    except:
        if verbose:
            print(f'Ошибка в мпстатс по ценам для этого sku ({sku}), данные отсутствуют.')
        
    sells.date = pd.to_datetime(sells.date)
    setts.date = pd.to_datetime(setts.date)
    res = setts.merge(sells, left_on=['date'], right_on=['date'], how='left')
    res['sku'] = sku
    res.fillna({'OrderQty': res.sales}, inplace=True)
    res['sales'] = res['OrderQty'].copy()
    cols_to_drop = ['SaleQty', 'OrderQty']
    cols_to_drop = [col for col in cols_to_drop if col in res.columns]
    res.drop(columns = cols_to_drop, inplace=True)
    res['sales'] = res['sales'].fillna(0)
    res['price'] = res['price'].bfill().ffill()
    res[['sales', 'price']] = res[['sales', 'price']].astype(int)
    return res

# sku = 52711454
# userid = '0517ba04-3f9d-4817-8d2c-96d1aacac050'
# date_from = (date.today() - timedelta(days=366)).isoformat(), 
# date_to = date.today().isoformat()

# Use case:
# data = get_data(sku)