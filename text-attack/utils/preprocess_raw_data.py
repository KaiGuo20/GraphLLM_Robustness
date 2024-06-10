import os
import sys
sys.path.append('')
import pickle
import numpy as np
import torch
from tqdm import tqdm
from utils.util import args
import argparse
import torch
from transformers import AutoTokenizer
from attacks.models import *
import attacks.models as models
from utils.util import *

#from utils.data import NewDataset
device = torch.device("cuda")

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
def write_pickle(filename, file):
    with open(filename, "wb") as f:
        return pickle.dump(file, f)
def load_file(filename):
    with open(filename, "rb") as f:
        return f.readlines()


data = torch.load(f"raw_data/{args.dataset.lower()}_{args.data_type}_sbert.pt")
print(data)
data.train_mask = data.train_masks[0]
data.val_mask = data.val_masks[0]
data.test_mask = data.test_masks[0]


model = models.BertC(dropout=0.5, num_class=7, device = device).to(device)
model.eval()
embeds = torch.FloatTensor([]).to(device)
for i in tqdm(range(30512)):
    embed = model.model.embeddings.word_embeddings(torch.LongTensor([i]).to(device))
    # print(embed.shape)
    embeds = torch.cat([embeds, embed])
print(embeds.shape)
torch.save(embeds, "processed_data/word_embeds.pt")

new = []

try:
    raw_text = data.raw_text
except:
    raw_text = data.raw_texts

for i, text in tqdm(enumerate(raw_text)):
    tmp = {"Node": i, "raw_text": text, "label": data.y[i].item()} # "category_name": data.category_names[i]
    new.append(tmp)
print(len(new))
write_pickle(f"processed_data/{args.dataset}/data.pkl", new)
print("done!")


