import numpy as np
import pandas as pd
import requests

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm.notebook import tqdm
from multiprocessing import cpu_count

from datasets import Dataset

L_BOUND = 0 # TODO: попробуем частями
R_BOUND = 1000
SAVE_NAME = 'test_1k'

def make_tg_report(text) -> None:
    token = '6498069099:AAFtdDZFR-A1h1F-8FvOpt6xIzqjCbdLdsc'
    method = 'sendMessage'
    chat_id = 324956476
    _ = requests.post(
            url='https://api.telegram.org/bot{0}/{1}'.format(token, method),
            data={'chat_id': chat_id, 'text': text}
        ).json()

def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers = {
                        "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
                        "X-Forwarded-For": "64.18.15.200",
                    }
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def fetch_images(batch, num_threads, timeout=2, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(tqdm(executor.map(fetch_single_image_with_args, batch["url"]), total=len(batch)))
    return batch

def main():
    path = 'Conceptual 12M russian.csv'
    conceptual_ds = Dataset.from_csv(path)
    conceptual_ds = conceptual_ds.select(range(L_BOUND, R_BOUND))
    num_threads = 4
    conceptual_ds = conceptual_ds.map(fetch_images, batched=True, 
                                      batch_size=100, fn_kwargs={"num_threads": num_threads})
    filtered = conceptual_ds.filter(lambda row: row['image'] is not None)
    filtered.save_to_disk(SAVE_NAME+'_filtered')
    # conceptual_ds.save_to_disk(SAVE_NAME)
    make_tg_report(SAVE_NAME + ' is ready')
    
if __name__ == '__main__':
    main()