# Parameters for main()
import os

DATA_PATH = 'data/'
TABLE_DATASET_FILE = 'new_labeled.csv'
IMG_DATASET_NAME = 'images_7k'
NAME_MODEL_NAME = 'DeepPavlov/distilrubert-tiny-cased-conversational-v1'
DESCRIPTION_MODEL_NAME = 'cointegrated/rubert-tiny'

# ------------------------------
# RuCLIPtiny: requires name_model_name explicitly
# ------------------------------
from timm import create_model
import torch
from torch import nn
from transformers import AutoModel
import tqdm

class RuCLIPtiny(nn.Module):
    def __init__(self, name_model_name: str):
        """
        Initializes the RuCLIPtiny module using the provided name model.
        """
        super().__init__()
        self.visual = create_model('convnext_tiny',
                                   pretrained=False,  # set True if you want pretrained weights
                                   num_classes=0,
                                   in_chans=3)       # output: e.g. 768-dim features
        
        self.transformer = AutoModel.from_pretrained(name_model_name)
        name_model_output_size = self.transformer.config.hidden_size  # inferred dynamically
        self.final_ln = nn.Linear(name_model_output_size, 768)         # project to 768 dims
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
    
    @property
    def dtype(self):
        return self.visual.stem[0].weight.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.type(self.dtype))

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # use the CLS token (first token)
        x = x.last_hidden_state[:, 0, :]
        x = self.final_ln(x)
        return x

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


# ------------------------------
# Tokenizers and Dataset: required name_model_name and description_model_name
# ------------------------------
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
import cv2
import os

def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class Tokenizers:
    def __init__(self, name_model_name: str, description_model_name: str):
        self.name_tokenizer = AutoTokenizer.from_pretrained(name_model_name)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(description_model_name)

    def tokenize_name(self, texts, max_len=77):
        tokenized = self.name_tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return torch.stack([tokenized["input_ids"], tokenized["attention_mask"]])

    def tokenize_description(self, texts, max_len=77):
        tokenized = self.desc_tokenizer(
            texts,
            truncation=True,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return torch.stack([tokenized["input_ids"], tokenized["attention_mask"]])

class SiameseRuCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, name_model_name: str, description_model_name: str, df=None, labels=None, df_path=None):
        """
        Dataset requires the concrete models' names for tokenization.
        """
        self.df = pd.read_csv(df_path) if df_path is not None else df
        self.labels = labels
        self.images_dir = images_dir
        self.tokenizers = Tokenizers(name_model_name, description_model_name)
        self.transform = get_transform()
        self.max_len = 77

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Tokenize names
        name_tokens = self.tokenizers.tokenize_name([str(row.name_first), str(row.name_second)], max_len=self.max_len)
        name_first = name_tokens[:, 0, :]  # [input_ids, attention_mask]
        name_second = name_tokens[:, 1, :]
        # Tokenize descriptions
        desc_tokens = self.tokenizers.tokenize_description([str(row.description_first), str(row.description_second)])
        desc_first = desc_tokens[:, 0, :]
        desc_second = desc_tokens[:, 1, :]
        # Process images
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

    def __len__(self):
        return len(self.df)


# ------------------------------
# SiameseRuCLIP: required models_dir, name_model_name, and description_model_name
# ------------------------------
from transformers import AutoModel

class SiameseRuCLIP(nn.Module):
    def __init__(self,
                 preload_ruclip: bool,
                 preload_model_name: str,
                 device: str,
                 models_dir: str,
                 name_model_name: str,
                 description_model_name: str):
        """
        Initializes the SiameseRuCLIP model.
        Required parameters:
          - models_dir: directory containing saved checkpoints.
          - name_model_name: model name for text (name) branch.
          - description_model_name: model name for description branch.
        """
        super().__init__()
        # Initialize RuCLIPtiny with the provided name model.
        self.ruclip = RuCLIPtiny(name_model_name)
        if preload_ruclip:
            std = torch.load(os.path.join(models_dir, preload_model_name),
                             weights_only=True,
                             map_location=device)
            self.ruclip.load_state_dict(std)
            self.ruclip = self.ruclip.to(device)
            self.ruclip.eval()
        # Initialize description transformer with the provided description model.
        self.description_transformer = AutoModel.from_pretrained(description_model_name)
        
        # Infer dimensions automatically from inner modules.
        vision_dim = self.ruclip.visual.num_features            # e.g. 768 from ConvNeXt tiny
        name_dim = self.ruclip.final_ln.out_features              # e.g. 768 after projection
        desc_dim = self.description_transformer.config.hidden_size  # e.g. 312 for cointegrated/rubert-tiny
        per_product_dim = vision_dim + name_dim + desc_dim         # total per–product embedding, e.g., 768+768+312 = 1848
        head_input_dim = 2 * per_product_dim                      # two products concatenated
        
        self.hidden_dim = per_product_dim
        
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Linear(head_input_dim // 2, head_input_dim // 4),
            nn.ReLU(),
            nn.Linear(head_input_dim // 4, 2)
        )
    
    def encode_description(self, desc):
        # desc: [input_ids, attention_mask]
        out = self.description_transformer(desc[:, 0, :], desc[:, 1, :])
        last_hidden = out.last_hidden_state
        attention_mask = desc[:, 1, :]
        # Average pooling over token representations.
        return (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    
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

from sklearn.metrics import f1_score

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

def plot_epoch(loss_history)->None:
    display.clear_output(wait=True)
    plt.figure(figsize=(8, 6))
    plt.title("Training loss")
    plt.xlabel("Iteration number")
    plt.ylabel("Loss")
    plt.plot(loss_history, 'b')
    plt.show()

def validation(model, valid_loader, score, device='cuda') -> float:
    correct_val = 0
    with torch.no_grad(): 
        model.eval()
        for data in tqdm(valid_loader):
            im1, name1, desc1, im2, name2, desc2, label = data 
            # Move all input tensors to the specified device
            im1, name1, desc1, im2, name2, desc2, label = im1.to(device), name1.to(device), desc1.to(device), im2.to(device), name2.to(device), desc2.to(device), label.to(device)  
            out = model(im1, name1, desc1, im2, name2, desc2) 
            _, predicted = torch.max(out.data, -1)
            predicted = predicted.cpu().numpy()
            # Move label to CPU and convert to NumPy array
            label = label.cpu().numpy()  # Added this line
            correct_val += score(label, predicted)
            # break
    return correct_val / len(valid_loader)

# ------------------------------
# Main Execution Code
# ------------------------------
def main():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader
    import torch
    from sklearn.metrics import f1_score
    import numpy as np
    import os

    labeled = pd.read_csv(DATA_PATH + TABLE_DATASET_FILE)

    images_dir = DATA_PATH + IMG_DATASET_NAME
    models_dir = DATA_PATH + 'trained_models/'
    
    BATCH_SIZE = 50
    EPOCHS = 15
    LR = 3e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_SPLITS = 3
    
    skf = StratifiedKFold(n_splits=N_SPLITS)
    nn_scores = []
    X, y = labeled.drop(columns='label'), labeled.label.values
    
    for train_index, valid_index in skf.split(X, y):
        train_df = X.iloc[train_index]
        train_labels = y[train_index]
        valid_df = X.iloc[valid_index]
        valid_labels = y[valid_index]
        
        train_dataset = SiameseRuCLIPDataset(df=train_df, labels=train_labels, images_dir=images_dir,
                                              name_model_name=NAME_MODEL_NAME,
                                              description_model_name=DESCRIPTION_MODEL_NAME)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        valid_dataset = SiameseRuCLIPDataset(df=valid_df, labels=valid_labels, images_dir=images_dir,
                                              name_model_name=NAME_MODEL_NAME,
                                              description_model_name=DESCRIPTION_MODEL_NAME)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
        
        # For this example, we assume no pretrained ruclip weights (preload_ruclip=False)
        model = SiameseRuCLIP(preload_ruclip=False,
                              preload_model_name='',  # not used when preload_ruclip is False
                              device=DEVICE,
                              models_dir=models_dir,
                              name_model_name=NAME_MODEL_NAME,
                              description_model_name=DESCRIPTION_MODEL_NAME)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        
        best_valid_score = train(model, optimizer, criterion,
                                 EPOCHS, train_loader, valid_loader,
                                 score=f1_score, device=DEVICE, print_epoch=True)
        nn_scores.append(best_valid_score)
    
    print("Mean validation score:", np.mean(nn_scores))

if __name__ == '__main__':
    main()