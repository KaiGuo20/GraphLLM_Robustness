import torch
from transformers import AutoTokenizer, AutoModel
from attacks import models
import ipdb
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import argparse
import random

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

def get_dict(similar_dict, tokenizer):
    new_dict = {0: [0], 101: [101]}
    for k, v in similar_dict.items():
        k = tokenizer.convert_tokens_to_ids(k)
        v = [tokenizer.convert_tokens_to_ids(x) for x in v]
        if k not in v:
            v.append(k)
        while 100 in v:
            v.remove(100)
        if len(v) >= 1:
            new_dict[k] = v
        else:
            new_dict[k] = [k]

    return new_dict


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora")
parser.add_argument('--model', type=str, default='sbert')
parser.add_argument('--data_type', type=str, default="fixed")
parser.add_argument('--finetune', type=bool, default=False)
parser.add_argument('--device', type=str, default="cuda")
args = parser.parse_args()


data_name = args.dataset
model_name = args.model
if args.finetune:
    ft_data = args.dataset
else:
    ft_data = None

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

dataset = torch.load(f"raw_data/{data_name}_{args.data_type}_sbert.pt")
try:
    dataset.raw_texts = dataset.raw_text
except:
    pass


if model_name == 'bow':
    feat_list = []
    clean_seqs = []
    candidates_list = []
    mod_nodes_list = []
    for split in range(5):
        test_nodes = torch.where(dataset.test_masks[split] == True)[0].numpy()
        mod_nodes = torch.LongTensor(random.sample(list(test_nodes), 200))
        mod_nodes_list.append(mod_nodes)
        text_encoder = models.BOW(dataset, 2048, split, device)
        seqs = []
        candidates = []
        new_feat = torch.FloatTensor([])
        for i, sent in enumerate(tqdm(dataset.raw_texts)):
            words = text_encoder.tokenizer(sent)
            seq = [text_encoder.vocab[word.lower()] for word in words]
            if len(seq) > 512:
                seq = seq[:512]
                words = words[:512]
            seqs.append(seq) 

            new_feat = torch.cat([new_feat, text_encoder.embedding.weight[seq].detach().cpu().sum(0).unsqueeze(0)])
            # ipdb.set_trace()
            if i in mod_nodes:
                knowledge_dict = {}
                for i, idx in enumerate(seq):
                    if idx > len(text_encoder.vocab) or idx == 0:
                        knowledge_dict[idx] = torch.LongTensor([idx])
                    else:
                        word_candidates = get_knowledge(words[i])
                        idx_candidates = [text_encoder.vocab[word] for word in word_candidates if word in text_encoder.words]
                        idx_candidates = torch.LongTensor(idx_candidates)
                        
                        if len(idx_candidates) == 0:
                            knowledge_dict[idx] = torch.LongTensor([idx])
                        else:
                            knowledge_dict[idx] = idx_candidates
            else:
                knowledge_dict = {}

            candidates.append(knowledge_dict)
        # ipdb.set_trace()
        feat_list.append(new_feat)
        clean_seqs.append(seqs)
        candidates_list.append(candidates)

        if args.dataset == "arxiv":
            dataset.x = new_feat
            dataset.seqs = seqs
            dataset.candidates = candidates
            dataset.mod_nodes = mod_nodes
            torch.save(dataset, f"raw_data/{data_name}_{args.data_type}_{model_name}_{split}.pt")


    dataset.clean_x = feat_list
    dataset.clean_seqs = clean_seqs
    dataset.clean_candidates = candidates_list
    dataset.mod_nodes_list = mod_nodes_list
    torch.save(dataset, f"raw_data/{data_name}_{args.data_type}_{model_name}.pt")

elif model_name=='llama' and data_name=='arxiv':
    text_encoder = models.BertC(model_name = model_name, dropout=0.5, num_class=7, ft_data = ft_data, device = device).to(device)
    seqs = torch.LongTensor([])
    new_feat = torch.FloatTensor([])
    seqs = text_encoder.tokenizer(dataset.raw_texts, truncation = True)['input_ids']
    for seq in tqdm(seqs):
        seq = torch.LongTensor(seq)
        new_feat = torch.cat([new_feat, text_encoder.generate(input_ids = seq.unsqueeze(0), seq_len = torch.LongTensor([len(seq)])).detach().cpu()])

    for split in range(5):
        test_nodes = torch.where(dataset.test_masks[split] == True)[0].numpy()
        mod_nodes = torch.LongTensor(random.sample(list(test_nodes), 200))
        candidates = []    
        for i, seq in tqdm(enumerate(seqs)):
            words = [text_encoder.tokenizer._convert_id_to_token(x) for x in seq]

            if i in mod_nodes:
                knowledge_dict = {}
                for i, idx in enumerate(seq):
                    word_candidates = get_knowledge(words[i])
                    idx_list = []
                    for word in word_candidates:
                        new_idx = text_encoder.tokenizer.convert_tokens_to_ids(word)
                        if new_idx >= 103:
                            idx_list.append(new_idx)
                    idx_list = torch.LongTensor(idx_list)
                    
                    if len(idx_list) == 0:
                        knowledge_dict[idx] = torch.LongTensor([idx])
                    else:
                        knowledge_dict[idx] = idx_list
            else:
                knowledge_dict = {}
            
            candidates.append(knowledge_dict)

        dataset.x = new_feat
        dataset.seqs = seqs
        dataset.candidates = candidates
        dataset.mod_nodes = mod_nodes
        if args.finetune:
            torch.save(dataset, f"raw_data/{data_name}_finetuned_{model_name}_{split}.pt")
        else:
            torch.save(dataset, f"raw_data/{data_name}_{args.data_type}_{model_name}_{split}.pt")

        

else:
    text_encoder = models.BertC(model_name = model_name, dropout=0.5, num_class=7, ft_data = ft_data, device = device).to(device)
    seqs = torch.LongTensor([])
    new_feat = torch.FloatTensor([])
    seqs = text_encoder.tokenizer(dataset.raw_texts, truncation = True)['input_ids']
    candidates = []
    
    for seq in tqdm(seqs):
        seq = torch.LongTensor(seq)
        new_feat = torch.cat([new_feat, text_encoder.generate(input_ids = seq.unsqueeze(0), seq_len = torch.LongTensor([len(seq)])).detach().cpu()])
        words = [text_encoder.tokenizer._convert_id_to_token(x) for x in seq]
        knowledge_dict = {}
        for i, idx in enumerate(seq):
            word_candidates = get_knowledge(words[i])
            idx_list = []
            for word in word_candidates:
                new_idx = text_encoder.tokenizer.convert_tokens_to_ids(word)
                if new_idx >= 103:
                    idx_list.append(new_idx)
            idx_list = torch.LongTensor(idx_list)
            
            if len(idx_list) == 0:
                knowledge_dict[idx.item()] = torch.LongTensor([idx])
            else:
                knowledge_dict[idx.item()] = idx_list
        
        candidates.append(knowledge_dict)
    # print(candidates)    
    # ipdb.set_trace() 
    dataset.seqs = seqs
    dataset.x = new_feat
    dataset.candidates = candidates
    if args.finetune:
        torch.save(dataset, f"raw_data/{data_name}_finetuned_{model_name}.pt")
    else:
        torch.save(dataset, f"raw_data/{data_name}_{args.data_type}_{model_name}.pt")


