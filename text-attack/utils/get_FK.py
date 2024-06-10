import sys
sys.path.append('')
import joblib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pytorch_transformers import BertTokenizer
from nltk.corpus import wordnet as wn
from collections import Counter
import os
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from time import time
from pandarallel import pandarallel
import pdb


from utils.util import args
from utils.util import load_pickle
import attacks.models as models

class NewDataset(Dataset):
    def __init__(self, path):
        self.max_len = 512
        cache_path = path.split(".pkl")[0] + '_all.pkl'
        if os.path.exists(cache_path):
            self.data = joblib.load(cache_path)
            save_path = cache_path
        else:
            self.data = joblib.load(path)
            save_path = path.split(".pkl")[0] + '_FK.pkl'
        # collect synonyms
        print("tokenizing")
        all_ids = set()
        for i, data in enumerate(tqdm(self.data)):
            data['seq'] = tokenizer(data['raw_text'])["input_ids"]
            if len(data['seq']) > self.max_len:
                data['seq'] = data['seq'][:self.max_len]
            data['seq_len'] = len(data['seq'])
            all_ids.update(data['seq'])
            tokenized_words = [tokenizer._convert_id_to_token(x) for x in data['seq']][1:]
        # print(all_ids)
        # df_words = pd.DataFrame({"words": all_words})
        # print("processing")
        # pandarallel.initialize(nb_workers=5)
        # df_words["words"].parallel_apply(preprocess_syn)
        print("preprocess finished")
        all_words = [tokenizer._convert_id_to_token(x) for x in all_ids]
        for i, ids in tqdm(enumerate(all_ids)):
            if all_words[i] not in syn_words:
                pdb.set_trace()
                words = get_knowledge(all_words[i])
                syn_words[all_words[i]] = words

        knowledge_data = []
        for i, data in enumerate(tqdm(self.data)):
            data['knowledge_dict'] = get_knowledge_dict(data['seq'])
            knowledge_data.append(data)
            if i % 1000 == 0:
                joblib.dump(knowledge_data, save_path)
        joblib.dump(self.data, save_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def preprocess_syn(word):
    if word not in syn_words:
        syn_words[word] = get_knowledge(word)
    return None


def transform(seq):
    
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    return tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(x) for x in seq])


def difference(a, b):
    tot = 0
    for x, y in zip(a, b):
        if x != y:
            tot += 1

    return tot


def get_knowledge(word):
    # pdb.set_trace()
    synset = wn.synsets(word)
    if len(synset) == 0:  # no synonym
        knowledge = [word]
        return knowledge
    else:
        new_synset = []
        for syn in synset:
            new_synset.append(syn.name().split('.')[0])
        synset = new_synset
        if word not in synset:
            synset.append(word)
        return list(set(synset))
    
    # ids = word
    # if ids <= 103:
    #     return []
    
    # word_embed = total_word_embeds[ids]

    # dist = torch.norm((word_embed - word_embeds), 2, -1)
    # M = torch.max(dist)
    # nei_ids = []
    # for i in range(400):
    #     idx = torch.argmin(dist)
    #     new_word_index = word_ids_list[idx]
    #     if new_word_index.item() > 103:
    #         nei_ids.append(new_word_index)
    #     dist[idx] = M
    #     if len(nei_ids) == 200:
    #         break
    # nei_words = tokenizer.convert_ids_to_tokens(nei_ids)
    # return nei_words

def get_knowledge_dict(indexed_tokens):
    knowledge_dict = {}
    tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
    for i in range(1, len(indexed_tokens)):
        # if tokenized_words[i] in word_list:  
        pdb.set_trace() 
        if tokenized_words[i] in syn_words:
            words = syn_words[tokenized_words[i]]
        else:
            
            words = get_knowledge(tokenized_words[i]) #  tokenized_words[i]  indexed_tokens[i]
            syn_words[tokenized_words[i]] = words
        # else:
        #     words = []
        if len(words) >= 1:
            knowledge_dict[tokenized_words[i]] = words
        else:
            knowledge_dict[tokenized_words[i]] = [tokenized_words[i]]

    return knowledge_dict


if __name__ == '__main__':
    if os.path.exists('processed_data/syn_words.pkl'):
        syn_words = joblib.load('processed_data/syn_words.pkl')
    else:
        syn_words = {}
    print(f"len syn words: {len(syn_words)}")
    
    word_embeds = torch.load("processed_data/word_embeds.pt")
    total_word_embeds = word_embeds.cuda()
    
    device = torch.device("cuda:1")
    
    # model = models.BertC(dropout=args.dropout, num_class=3, device = device).to(device)
    tokenizer = AutoTokenizer.from_pretrained('Sbert', padding=True, truncation=True, return_tensors="pt")

    word_list = list(set(load_pickle("raw_data/word_list.pkl")))
    word_ids_list = torch.LongTensor(tokenizer.convert_tokens_to_ids(word_list)).cuda()
    word_embeds = total_word_embeds[word_ids_list]
    total_words = set()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    test_data = NewDataset(f"processed_data/{args.dataset}/data.pkl")
    joblib.dump(syn_words, 'processed_data/syn_words.pkl')

