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

from utils.util import args
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
        knowledge_data = []
        for i, data in enumerate(tqdm(self.data)):
            if 'seq' not in data:
                data['seq'] = tokenizer(data['raw_text'])["input_ids"]
                if len(data['seq']) > self.max_len:
                    data['seq'] = data['seq'][:self.max_len]
                data['seq_len'] = len(data['seq'])
            data['knowledge_dict'] = get_knowledge_dict(data['seq'])
            knowledge_data.append(data)
            if i % 1000 == 0:
                joblib.dump(knowledge_data, save_path)
        joblib.dump(self.data, save_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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
    knowledge = [word]
    synset = wn.synsets(word)
    hyposet = []
    hyposet += synset
    for item in synset:
        hyposet += item.hyponyms()
        hyposet += item.hypernyms()
    if len(synset) == 0:  # no synonym
        return knowledge
    else:
        posset = [syn.name().split('.')[1] for syn in synset]  # pos set
        pos = Counter(posset).most_common(1)[0][0]  # most common pos in synset
        new_synset = []
        for syn in synset:
            if syn.name().split('.')[1] == pos:  # only choose synonyms with the most common pos
                new_synset.append(syn.name().split('.')[0])
        synset = new_synset
        if word not in synset:
            synset.append(word)
        return list(set(synset))


def get_knowledge_dict(indexed_tokens):
    knowledge_dict = {}
    tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
    for i in range(1, len(indexed_tokens)):
        if tokenized_words[i] in word_list:
            words = get_knowledge(tokenized_words[i])
        else:
            words = []
        if len(words) >= 1:
            knowledge_dict[tokenized_words[i]] = words
        else:
            knowledge_dict[tokenized_words[i]] = [tokenized_words[i]]

    return knowledge_dict


if __name__ == '__main__':
    device = torch.device("cuda")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = models.BertC(dropout=args.dropout, num_class=7, device = device).to(device)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    word_list = joblib.load(args.word_list)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    test_data = NewDataset(args.test_data)

