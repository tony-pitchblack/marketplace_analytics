from timm import create_model
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch import optim, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary
from transformers import AutoModel, AutoTokenizer
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

import cv2

from PIL import Image
from tqdm.auto import tqdm

import requests

from io import BytesIO
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from IPython import display

DATA_PATH = 'data/'
NAME_MODEL_NAME = 'DeepPavlov/distilrubert-tiny-cased-conversational-v1'
DESCRIPTION_MODEL_NAME = 'cointegrated/rubert-tiny' # TODO sergeyzh/rubert-tiny-turbo deepvk/USER-base
SEED = 42

class Predictor:
    def __init__(self, model, images_dir=DATA_PATH+'images/', bs=16, device='cpu'):
        self.model = model
        self.model.eval()
        self.bs = bs
        self.device = device
        self.images_dir = images_dir

    def __call__(self, X, y):
        ds = SiameseRuCLIPDataset(self.images_dir, None, X, y)
        dl = DataLoader(ds, batch_size=self.bs)
        
        labels = list()
        probs = list()
        with torch.no_grad(): 
            for data in dl:
                im1, ii1, am1, im2, ii2, am2, label = data 
                im1, ii1, am1, im2, ii2, am2, label = im1.to(self.device), ii1.to(self.device), am1.to(self.device), im2.to(self.device), \
                        ii2.to(self.device), am2.to(self.device), label.to(self.device)
                out = self.model(im1, ii1, am1, im2, ii2, am2) 
                probs_, labels_ = torch.max(out.data.softmax(dim=1), -1)
                labels.extend(labels_.numpy().tolist())
                probs.extend(probs_.numpy().tolist())
        return np.array(labels), np.array(probs)
    
class RuCLIPtiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = create_model('convnext_tiny',
                                   pretrained=False, # TODO: берём претрейн
                                   num_classes=0,
                                   in_chans=3)  # out 768
        # text_config = DistilBertConfig(**{"vocab_size": 30522,
        #                                   "max_position_embeddings": 512,
        #                                   "n_layers": 3,
        #                                   "n_heads": 12,
        #                                   "dim": 264,
        #                                   "hidden_dim": 792,
        #                                   "model_type": "distilbert"})
        # self.transformer = DistilBertModel(text_config)
        self.transformer = AutoModel.from_pretrained(NAME_MODEL_NAME) # 312
        self.final_ln = torch.nn.Linear(312, 768) # 312 -> 768
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.stem[0].weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.final_ln(x)
        return x

    def forward(self, image, input_ids, attention_mask):
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
    

class Tokenizers:
    def __init__(self):
        self.name_tokenizer = AutoTokenizer.from_pretrained(NAME_MODEL_NAME)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(DESCRIPTION_MODEL_NAME)

    def tokenize_name(self, texts, max_len=77):
        tokenized = self.name_tokenizer.batch_encode_plus(texts,
                                                     truncation=True,
                                                     add_special_tokens=True,
                                                     max_length=max_len,
                                                     padding='max_length',
                                                     return_attention_mask=True,
                                                     return_tensors='pt')
        return torch.stack([tokenized["input_ids"], tokenized["attention_mask"]])
    
    def tokenize_description(self, texts, max_len=77):
        tokenized = self.desc_tokenizer(texts,
                                        truncation=True,
                                        add_special_tokens=True,
                                        max_length=max_len,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        return_tensors='pt')
        return torch.stack([tokenized["input_ids"], tokenized["attention_mask"]])
    
def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]), ])

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def download_images(df, save_path=''):
    def get_sku_image(url):
        img_data = requests.get(url).content
        try:
            img_data = Image.open(BytesIO(img_data))
        except: 
            img_data = None
        return img_data
    
    for sku_first in tqdm(df['sku_first'].unique()):
        temp = df[df['sku_first'] == sku_first].copy()
        for row in temp.iterrows():
            row = row[1]
            img1 = get_sku_image(row.image_url_first)
            img2 = get_sku_image(row.image_url_second)
            if img1 is not None and img2 is not None:
                img1.save(save_path + str(row.sku_first) + '.jpg')
                img2.save(save_path + str(row.sku_second) + '.jpg')

class SiameseRuCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df=None, labels=None, df_path=None, images_dir=DATA_PATH+'images/'):
        # loads data either from path using `df_path` or directly from `df` argument
        self.df = pd.read_csv(df_path) if df_path is not None else df
        self.labels = labels
        self.images_dir = images_dir
        self.tokenizers = Tokenizers()
        self.transform = get_transform()
        # 
        self.max_len = 77
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name_tokens = self.tokenizers.tokenize_name([str(row.name_first), 
                                               str(row.name_second)], max_len=self.max_len)
        name_first = name_tokens[:, 0, :] # [input_ids, attention_mask]
        name_second = name_tokens[:, 1, :]
        desc_tokens = self.tokenizers.tokenize_description([str(row.description_first), 
                                               str(row.description_second)])
        desc_first = desc_tokens[:, 0, :] # [input_ids, attention_mask]
        desc_second = desc_tokens[:, 1, :]
        im_first = cv2.imread(os.path.join(self.images_dir, row.image_name_first))
        im_first = cv2.cvtColor(im_first, cv2.COLOR_BGR2RGB)
        im_first = Image.fromarray(im_first)
        im_first = self.transform(im_first)
        im_second = cv2.imread(os.path.join(self.images_dir, row.image_name_second))
        im_second = cv2.cvtColor(im_second, cv2.COLOR_BGR2RGB)
        im_second = Image.fromarray(im_second)
        im_second = self.transform(im_second)
        label = self.labels[idx]
        return im_first, name_first, desc_first, im_second, name_second, desc_second, label

    def __len__(self,):
        return len(self.df)

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
class SiameseRuCLIP(nn.Module):
    def __init__(self,
                 preload_ruclip=False,
                 preload_model_name='cc12m_rubert_tiny_ep_1.pt', # 'cc12m_ddp_4mill_ep_4.pt'
                 device='cpu',
                 models_dir=DATA_PATH + 'train_results/'):
        super().__init__()
        self.ruclip = RuCLIPtiny()

        if preload_ruclip:
            std = torch.load(
                models_dir + preload_model_name,
                weights_only=True,
                map_location=device
            )
            self.ruclip.load_state_dict(std)
            self.ruclip = self.ruclip.to(device)
            self.ruclip.eval()

        self.description_transformer = AutoModel.from_pretrained(DESCRIPTION_MODEL_NAME)

        # Automatically infer dimensions:
        # For the vision encoder, we assume the timm model provides 'num_features'
        vision_dim = self.ruclip.visual.num_features  # e.g. 768 for convnext_tiny
        
        # For the name branch, use the output dimension of the final linear layer.
        name_dim = self.ruclip.final_ln.out_features   # e.g. 768
        
        # For the description transformer, take the hidden size from its configuration.
        desc_dim = self.description_transformer.config.hidden_size  # e.g. 312 for cointegrated/rubert-tiny
        
        # Compute the per-product embedding as the concatenation of the three modalities.
        per_product_dim = vision_dim + name_dim + desc_dim  # e.g. 768 + 768 + 312 = 1848
        head_input_dim = 2 * per_product_dim  # for a pair of products
        
        self.hidden_dim = per_product_dim  # storing the per-product dimension
        
        # Build the MLP head that takes concatenated features from two products.
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Linear(head_input_dim // 2, head_input_dim // 4),
            nn.ReLU(),
            nn.Linear(head_input_dim // 4, 2)
        )
        
    def encode_description(self, desc):
        # desc is [input_ids, attention_mask]
        last_hidden_states = self.description_transformer(desc[:, 0, :], desc[:, 1, :]).last_hidden_state
        attention_mask = desc[:, 1, :]
        # TODO: нужно ли делать пулинг, посмотреть на результаты
        return average_pool(last_hidden_states, attention_mask)
    
    def forward(self, im1, name1, desc1, im2, name2, desc2):
        image_emb1 = self.ruclip.encode_image(im1)
        image_emb2 = self.ruclip.encode_image(im2)
        name_emb1 = self.ruclip.encode_text(name1[:, 0, :], name1[:, 1, :])
        name_emb2 = self.ruclip.encode_text(name2[:, 0, :], name2[:, 1, :])
        desc_emb1 = self.encode_description(desc1) 
        desc_emb2 = self.encode_description(desc2)
        first_emb = torch.cat([image_emb1, name_emb1, desc_emb1], dim=1)
        second_emb = torch.cat([image_emb2, name_emb2, desc_emb2], dim=1)
        x = torch.cat([first_emb, second_emb], dim=1)
        out = self.head(x)
        return out

def train(model, optimizer, criterion, 
          epochs_num, train_loader, valid_loader=None, 
          score=f1_score, device='cpu', print_epoch=False) -> None:
    model.train()
    counter = []
    loss_history = [] 
    it_number = 0
    best_valid_score = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                               factor=0.1, patience=2,
                                                               threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08)
    for epoch in range(epochs_num):
        print("Epoch：", epoch)
        for i, data in enumerate(tqdm(train_loader)):
            im1, name1, desc1, im2, name2, desc2, label = data 
            im1, name1, desc1, im2, name2, desc2, label = im1.to(device), name1.to(device), desc1.to(device), im2.to(device), name2.to(device), desc2.to(device), label.to(device)
            optimizer.zero_grad() 
            out = model(im1, name1, desc1, im2, name2, desc2)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            if i % 5 == 0: # show changes of loss value after each 10 batches
                it_number += 5
                counter.append(it_number)
                loss_history.append(loss.item())
        # test after each epoch
        if print_epoch:
            valid_score = validation(model, valid_loader, score, device)
            plot_epoch(loss_history)
            print(f'Current loss: {loss}')
            print(f'Current {score.__name__}: {valid_score}')
            scheduler.step(valid_score)
            if valid_score > best_valid_score:
                best_valid_score = valid_score
    return best_valid_score

def validation(model, valid_loader, score, device='cpu') -> float:
    correct_val = 0
    with torch.no_grad(): 
        model.eval()
        for data in tqdm(valid_loader):
            im1, name1, desc1, im2, name2, desc2, label = data 
            im1, name1, desc1, im2, name2, desc2 = im1.to(device), name1.to(device), desc1.to(device), im2.to(device), name2.to(device), desc2.to(device)
            out = model(im1, name1, desc1, im2, name2, desc2) 
            _, predicted = torch.max(out.data, -1)
            predicted = predicted.cpu().numpy()
            correct_val += score(label, predicted)
    return correct_val / len(valid_loader)

def plot_epoch(loss_history)->None:
    display.clear_output(wait=True)
    plt.figure(figsize=(8, 6))
    plt.title("Training loss")
    plt.xlabel("Iteration number")
    plt.ylabel("Loss")
    plt.plot(loss_history, 'b')
    plt.show()
    
def make_tg_report(text) -> None:
    token = '6498069099:AAFtdDZFR-A1h1F-8FvOpt6xIzqjCbdLdsc'
    method = 'sendMessage'
    chat_id = 324956476
    _ = requests.post(
            url='https://api.telegram.org/bot{0}/{1}'.format(token, method),
            data={'chat_id': chat_id, 'text': text} 
        ).json()
                
def main():
    labeled = pd.read_csv(DATA_PATH + 'labeled.csv')
    got_images = True
    images_dir = DATA_PATH + 'images_5k/'
    if not got_images:
        download_images(labeled, images_dir)
        
    BATCH_SIZE=50
    EPOCHS=15
    LR=3e-5
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    N_SPLITS=3
    
    skf = StratifiedKFold(n_splits=N_SPLITS)
    nn_scores = list()
    X, y = labeled.drop(columns='label'), labeled.label.values
    for train_index, valid_index in skf.split(X, y): # разбивка по индексам
        train_dataset = SiameseRuCLIPDataset(X.iloc[train_index], y[train_index], images_dir=images_dir)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        valid_dataset = SiameseRuCLIPDataset(X.iloc[valid_index], y[valid_index], images_dir=images_dir)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
        
        model = SiameseRuCLIP(device=DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        best_valid_score = train(model, optimizer, criterion,
            EPOCHS, train_loader, valid_loader, 
            score=f1_score, print_epoch=True)
        nn_scores.append(best_valid_score)

    make_tg_report(str(np.mean(nn_scores)))