import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import mlflow
from mlflow.models import infer_signature

from timm import create_model
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchinfo import summary
import transformers
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer,\
        get_linear_schedule_with_warmup

import cv2

from PIL import Image
from tqdm.auto import tqdm

import json
from itertools import product

class Tokenizer:
    def __init__(self):
        tokenizer_load = "DeepPavlov/distilrubert-tiny-cased-conversational-v1"
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_load)

    def tokenize(self, texts, max_len=77):
        tokenized = self.tokenizer.batch_encode_plus(texts,
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


class RuCLIPTinyDataset(Dataset):
    def __init__(self, dir, df_path, max_text_len=77):
        self.df = pd.read_csv(df_path)
        # self.df = self.df[:len(self.df)//8] # TODO: взял четверть чтобы потестить параметры
        self.dir = dir
        self.max_text_len = max_text_len
        self.tokenizer = Tokenizer()
        self.transform = get_transform()

    def __getitem__(self, idx):
        # достаем имя изображения и ее лейбл
        image_name = self.df['image_path'].iloc[idx]
        text = self.df['text'].iloc[idx]
        tokens = self.tokenizer.tokenize([text], max_len=self.max_text_len)
        input_ids, attention_mask = tokens[0][0], tokens[1][0]
        image = cv2.imread(os.path.join(self.dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, input_ids, attention_mask

    def __len__(self):
        return len(self.df)


class RuCLIPtiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = create_model('convnext_tiny',
                                   pretrained=True, # TODO: берём претрейн
                                   num_classes=0,
                                   in_chans=3)  # out 768
        text_config = DistilBertConfig(**{"vocab_size": 30522,
                                          "max_position_embeddings": 512,
                                          "n_layers": 3,
                                          "n_heads": 12,
                                          "dim": 264,
                                          "hidden_dim": 792,
                                          "model_type": "distilbert"})
        self.transformer = DistilBertModel(text_config)
        self.final_ln = torch.nn.Linear(264, 768)
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

class Predictor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.transform = get_transform()

    def prepare_images_features(self, model, images, device='cpu'):
        images_features = []
        for image in images:
            # image = Image.open(image_path)
            image = self.transform(image)
            with torch.no_grad():
                image_features = model.encode_image(image.unsqueeze(0).to(device)).float().cpu()[0]
            images_features.append(image_features)
        images_features = torch.stack(images_features, axis=0)
        images_features /= images_features.norm(dim=-1, keepdim=True)
        return images_features.cpu()

    def prepare_text_features(self, model, texts, max_len=77, device='cpu'):
        texts_features = []
        for text in texts:
            tokens = self.tokenizer.tokenize([text], max_len)
            with torch.no_grad():
                text_features = model.encode_text(tokens[0].to(device), tokens[1].to(device)).float().cpu()[0]
            texts_features.append(text_features)
        texts_features = torch.stack(texts_features, axis=0)
        texts_features /= texts_features.norm(dim=-1, keepdim=True)
        return texts_features

    def __call__(self, model, images, classes, get_probs=False, max_len=77, device='cpu'):
        model.eval().to(device)
        image_features = self.prepare_images_features(model, images, device)
        texts_features = self.prepare_text_features(model, classes, max_len, device)
        text_probs = (1 * image_features @ texts_features.T).softmax(dim=-1)
        if get_probs:
            return text_probs
        else:
            return text_probs.argmax(-1)
        
class Trainer:
    def __init__(self, train_dataframe, train_dir,
                 val_dataframe=None, val_dir=None, learning_rate=5e-5,
                 freeze_image_encoder=True, freeze_text_encoder=False, max_text_len=77,
                 train_batch_size=64, val_batch_size=64, num_workers=2,
                 weight_decay=1e-4, grad_accum=8,
                 num_warmup_steps=0,
                 save_name='best_model_vis_fr',
                 optimizer=None):
        self.train_dataframe = train_dataframe
        self.train_dir = train_dir
        self.val_dataframe = val_dataframe
        self.val_dir = val_dir
        self.learning_rate = learning_rate
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_text_encoder = freeze_text_encoder
        self.max_text_len = max_text_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.grad_accum = grad_accum
        self.num_warmup_steps = num_warmup_steps
        self.save_name=save_name
        self.optimizer=optimizer
        print(f"train batch size = {self.train_batch_size * self.grad_accum}")

    def train_model(self, model, epochs_num=1, device='cuda', verbose=10):

        is_val = self.val_dataframe is not None and self.val_dir is not None

        model.to(device)

        train_dataset = RuCLIPTinyDataset(self.train_dir, self.train_dataframe, self.max_text_len)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=self.train_batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=self.num_workers)

        if is_val:
            val_dataset = RuCLIPTinyDataset(self.val_dir, self.val_dataframe, self.max_text_len)
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=self.val_batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=self.num_workers)

        for i, child in enumerate(model.children()):
            if (i == 0 and self.freeze_image_encoder) or (i == 1 and self.freeze_text_encoder):
                for param in child.parameters():
                    param.requires_grad = False
        model.visual.stages[3].blocks[2].mlp.fc2.requires_grad = True # TODO: последний слой размораживаем
        
        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()
        if self.optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-8,
                                          weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = len(train_loader) * epochs_num
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                                 num_warmup_steps=self.num_warmup_steps, # int(0.20 * total_steps) # TODO: нет разогрева
        #                                                 num_training_steps=total_steps)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                               factor=0.1, patience=1,
                                                               verbose=True, threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08)
        # mlflow pipeline
        params = {
            "epochs": epochs_num,
            "learning_rate": self.learning_rate,
            "batch_size": self.train_batch_size,
            'grad_accum': self.grad_accum, 
            'weight_decay': self.weight_decay, 
            'num_workers': self.num_workers,
            "loss_function_img": loss_img.__class__.__name__,
            "loss_function_txt": loss_txt.__class__.__name__,
            'scheduler': type(scheduler).__name__,
            'num_warmup_steps': self.num_warmup_steps, 
            "optimizer": type(self.optimizer).__name__,
        }
        mlflow.log_params(params)
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")
        ###########
        best_loss = 1e5
        for epoch in range(epochs_num):
            model.train()
            print(f'start training epoch {epoch}')
            curr_batch = 0
            X = []
            Y = []
            curr_batch = 0
            for i, batch in enumerate(tqdm(train_loader)):
                images = batch[0].cuda()
                input_ids = batch[1].cuda()
                attention_mask = batch[2].cuda()

                image_features = model.encode_image(images)
                text_features = model.encode_text(input_ids, attention_mask)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                X.append(image_features)
                Y.append(text_features)

                if ((i + 1) % self.grad_accum == 0) or (i + 1 == len(train_loader)):
                    optimizer.zero_grad()
                    X = torch.cat(X, axis=0).cuda()
                    Y = torch.cat(Y, axis=0).cuda()
                    logit_scale = model.logit_scale.exp()
                    logits_per_image = logit_scale * X @ Y.t()
                    logits_per_text = logits_per_image.t()
                    ground_truth = torch.arange(X.shape[0], dtype=torch.long).cuda()
                    img_l = loss_img(logits_per_image, ground_truth)
                    text_l = loss_txt(logits_per_text, ground_truth)
                    total_loss = (img_l + text_l) / 2
                    if curr_batch % verbose == 0:
                        print(f'{i}/{len(train_loader)} total_loss {total_loss}')
                        step = i // 100 * (epoch + 1)
                        mlflow.log_metric("total_loss", f"{total_loss:2f}", step=step)
                    total_loss.backward()   
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    # scheduler.step()
                    
                    X = []
                    Y = []
                    curr_batch += 1
            if is_val:
                print(f'start val epoch {epoch}')
                total_loss = 0
                model.eval()
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(val_loader)):
                        images = batch[0].to(device)
                        input_ids = batch[1].to(device)
                        attention_mask = batch[2].to(device)

                        logits_per_image, logits_per_text = model(images, input_ids, attention_mask)
                        ground_truth = torch.arange(batch[1].shape[0], dtype=torch.long).to(device)
                        img_l = loss_img(logits_per_image, ground_truth).item()
                        text_l = loss_txt(logits_per_text, ground_truth).item()
                        total_loss += (img_l + text_l) / 2
                    total_loss /= len(val_loader)
                    print(f'val loss = {total_loss}')
                    mlflow.log_metric("val_loss", f"{total_loss:2f}", step=epoch)
                    lr_to_log = optimizer.param_groups[0]['lr']
                    mlflow.log_metric("lr", f"{lr_to_log:2f}", step=epoch)
                    scheduler.step(total_loss)
                if total_loss < best_loss:
                    best_loss = total_loss
                    torch.save(model.state_dict(), self.save_name + f'_ep_{epoch}' + ".pt")
        # return model



def main(epochs_num, learning_rate, weight_decay, batch_size, grad_accum, save_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    coco_path = 'coco2017/'
    train_dir = coco_path + 'train2017'
    val_dir = coco_path + 'val2017'
    coco_dfs = coco_path + 'captions_ru/'
    train_dataframe = coco_dfs + 'train_df.csv'
    val_dataframe = coco_dfs + 'val_df.csv'
    
    # mlflow pipeline
    mlflow.set_tracking_uri(uri="http://192.168.88.61:5000")
    mlflow.set_experiment("ruclip_tiny")
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run():
        trainer = Trainer(train_dataframe=train_dataframe, train_dir=train_dir,
                        val_dataframe=val_dataframe, val_dir=val_dir,
                        train_batch_size=batch_size, grad_accum=grad_accum, num_workers=4,
                        save_name=save_name,
                        learning_rate=learning_rate, optimizer=None, weight_decay=weight_decay)
        model = RuCLIPtiny().to(device)
        trainer.train_model(model, epochs_num=epochs_num, device=device, verbose=2)
        mlflow.pytorch.log_model(model, "model")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ruclip tiny train pipeline')
    parser.add_argument('epochs_num', type=int, help='Total epochs to train the model')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Input batch size on each device (default: 64)')
    parser.add_argument('--learning_rate', default=9e-4, type=float, help='Learning rate (default: 9e-4)')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay for optimizer (default: 1e-3)')
    parser.add_argument('--grad_accum', default=32, type=int, help='Num of batches to accumulate grad (default: 32)')
    parser.add_argument('--save_name', default='best_model', type=str, help='Name to save the model (default: best_model)')
    args = parser.parse_args()
    
    main(**vars(args))
    
    
    
    
    
    
    # def grid(grid_values):
    # for p in grid_values:
    #     # Always sort the keys of a dictionary, for reproducibility
    #     items = sorted(p.items())
    #     if not items:
    #         yield {}
    #     else:
    #         keys, values = zip(*items)
    #         for v in product(*values):
    #             params = dict(zip(keys, v))
    #             yield params
    
    # if grid_search:
    # grid_values= [{"learning_rate": [3e-4, 6e-4, 9e-4], 
    #                "optimizer": [optim.SGD, optim.AdamW],
    #                'weight_decay':[1e-3, 1e-4]}]
    # for params in grid(grid_values):
    #     mlflow.set_experiment("ruclip_gridsearch")
    #     mlflow.enable_system_metrics_logging()
    #     learning_rate = params['learning_rate']
    #     optimizer = params['optimizer']
    #     weight_decay = params['weight_decay']
    #     with mlflow.start_run():
    #         trainer = Trainer(train_dataframe=train_dataframe, train_dir=train_dir,
    #                         val_dataframe=val_dataframe, val_dir=val_dir,
    #                         train_batch_size=batch_size, grad_accum=grad_accum, num_workers=4,
    #                         save_name=save_name,
    #                         learning_rate=learning_rate, optimizer=optimizer, weight_decay=weight_decay)
    #         model = RuCLIPtiny().to(device)
    #         trainer.train_model(model, epochs_num=epochs_num, device=device, verbose=2)
    #         mlflow.pytorch.log_model(model, "model")