import torch
import os.path as osp
from data import get_tf_idf_by_texts, get_Anglellama_embedding,get_bow, get_llama_ft_embedding, get_llm2vec_embedding,get_deberta_embedding, get_llama_embedding, get_uae_embedding, get_T5_embedding, get_word2vec, get_sbert_embedding, set_api_key, get_ogbn_dataset, get_e5_large_embedding, get_ada_embedding_002_embedding, get_palm_cortex_001_embedding
from api import openai_ada_api
import h5py
import numpy as np
from torch_geometric.utils import index_to_mask
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from data import set_seed_config, LabelPerClassSplit, generate_random_mask
from utils import knowledge_augmentation
import chardet
import pickle
import dgl

import pandas as pd

def read_data(filename):
    df = pd.read_csv(filename)
    return df

def main():
    # dataset = ['cora', 'pubmed', 'citeseer', 'arxiv']
    dataset = ['wikics']
    split = ['fixed']
    ogb_dataset = ['arxiv', 'products']
    # embedding = ["know_inp_ft", "know_sep_ft", "tfidf", "word2vec", "sbert", "know_inp_sb", "know_sep_sb", "ada", "llama"]
    embedding = ["bow"]#palm-cortex-001
    # knowledge = ["cora", "pubmed"]
    data_path = "./preprocessed_data"
    ## if match default, just skip
    default = {
        'cora': 'sbert',
        "citeseer": 'llama',
        "pubmed": 'know_exp_e5',
        "arxiv": 'palm-cortex-001',#text-ada-embedding-002， palm-cortex-001
        "products": 'bow',
        "wikics": 'llm2vec',
        "citeseer2": 'llama_ft',
        "history": 'bow',
        "ptbcora": 'bow',
        "ptbarxiv": 'tfidf',
        "ptbciteseer2": 'e5',
        "ptbwikics": 'bow',
        "ptbpubmed": 'bow',
        "ptbhistory": 'tfidf'
    }
    split_seeds = [i for i in range(10)]
    ## load raw text data
    ## handle mask issue
    data_obj = None
    for name in dataset:
        for setting in split:
            if name in ogb_dataset and setting == 'random': continue
            if name == "cora" and setting == 'random':
                data_obj = torch.load("./preprocessed_data/new/cora_random_sbert.pt", map_location="cpu")
                data_obj.raw_texts = data_obj.raw_text
                data_obj.category_names = [data_obj.label_names[i] for i in data_obj.y.tolist()]
            elif name == "cora" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/cora_fixed_sbert.pt", map_location="cpu")
                data_obj.raw_texts = data_obj.raw_text
                data_obj.category_names = [data_obj.label_names[i] for i in data_obj.y.tolist()]
            elif name == "ptbcora" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/ptbcora_fixed_sbert.pt", map_location="cpu")
                embedding_str = embedding[0]
                path = "./preprocessed_data/new/cora_fixed_" + embedding_str + ".pt"
                data_old = torch.load(path, map_location="cpu")
                data_obj.x = data_old.x
                print('0000', data_obj.x.shape)
            elif name == "ptbciteseer2" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/ptbciteseer2_fixed_sbert.pt", map_location="cpu")
                embedding_str = embedding[0]
                path = "./preprocessed_data/new/citeseer2_fixed_" + embedding_str + ".pt"
                data_old = torch.load(path, map_location="cpu")
                data_obj.x = data_old.x
            elif name == "ptbpubmed" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/ptbpubmed_fixed_sbert.pt", map_location="cpu")
                embedding_str = embedding[0]
                path = "./preprocessed_data/new/pubmed_fixed_" + embedding_str + ".pt"
                data_old = torch.load(path, map_location="cpu")
                data_obj.x = data_old.x
            elif name == "citeseer" and setting == 'random':
                data_obj = torch.load("./preprocessed_data/new/citeseer_random_sbert.pt", map_location="cpu")
            elif name == "wikics" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/wikics_fixed_sbert.pt", map_location="cpu")
            elif name == "citeseer2" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/citeseer2_fixed_sbert.pt", map_location="cpu")
            elif name == "history" and setting == 'fixed':
                data_obj = dgl.load_graphs('./preprocessed_data/new/history_fixed_sbert.pt')
                print(data_obj)
                print(data_obj[0][0].edata['edge_attr'])
                text = pd.read_csv('./preprocessed_data/new/History_Final.csv')
                text = text['text'].tolist()
                print(text[0])
            elif name == "photo" and setting == 'fixed':
                # data_obj = torch.load("./preprocessed_data/new/history_fixed_sbert.pt", map_location="cpu")
                data_obj1 = torch.load("./preprocessed_data/new/cora_fixed_sbert.pt", map_location="cpu")
                print(data_obj1)
                data_obj = dgl.load_graphs('./preprocessed_data/new/photo_fixed_sbert.pt')
                print(data_obj)
                print('ndata', data_obj[0][0].ndata["feat"])
                text = pd.read_csv('./preprocessed_data/new/History_Final.csv')
                text = text['text'].tolist()
                print(text[0])
            elif name == "pubmed" and setting == 'random':
                data_obj = torch.load("./preprocessed_data/new/pubmed_random_sbert.pt", map_location="cpu")
            elif name == "pubmed" and setting == 'fixed':
                data_obj = torch.load("./preprocessed_data/new/pubmed_fixed_sbert.pt", map_location="cpu")
            elif name == "arxiv":
                print('arxiv')
                data_obj = torch.load("./preprocessed_data/new/arxiv_fixed_sbert.pt", map_location="cpu")
            elif name == "products":
                data_obj = torch.load("./preprocessed_data/new/products_fixed_sbert.pt", map_location="cpu")
            if name == 'cora' or name == 'pubmed':
                #d_name = name.split("_")[0]
                d_name = name
                entity_pt = torch.load(f"{d_name}_entity.pt", map_location="cpu")
                data_obj = torch.load(osp.join(data_path, "new", f"{d_name}_fixed_sbert.pt"), map_location="cpu")
                data_obj.entity = entity_pt
            print('data----', data_obj)
            # print('data.raw', data_obj.raw_texts)
            if name.startswith("ptb"):
                num_nodes = len(data_obj.ptb_texts[0])
                print('ptb-pub node', num_nodes)
            hidden_dim = 768
            for typ in embedding:
                if osp.exists(osp.join(data_path, "new", f"{name}_{setting}_{typ}.pt")):
                    print('--exist---')

                if default[name] == typ:
                    if typ == 'tfidf':
                        print('-----------')
                        if name == 'cora' or name == 'ptbcora':
                            max_features = 1433
                        elif name == 'citeseer':
                            max_features = 3703
                        elif name == 'pubmed' or name == 'ptbpubmed':
                            max_features = 500
                        else:
                            max_features = 1000
                        if name.startswith("ptb"):
                            data_obj.x_mix = [[] for _ in range(5)]
                            for i in range(5):
                                ptb_text = data_obj.ptb_texts[i] + data_obj.raw_texts
                                x, _ = get_tf_idf_by_texts(ptb_text, None, None, max_features=max_features,
                                                           use_tokenizer=False)
                                data_obj.ptb_features[i] = x[:num_nodes]
                                data_obj.x_mix[i] = x[num_nodes:]
                        else:
                            data_obj.x, _ = get_tf_idf_by_texts(data_obj.raw_texts, None, None, max_features=max_features, use_tokenizer=False)
                        # print('------------data_obj.x', data_obj.x)
                    elif typ == 'know_tf':
                        if name == 'cora':
                            max_features = 1433
                        elif name == 'citeseer':
                            max_features = 3703
                        elif name == 'pubmed':
                            max_features = 500
                        texts, knowledge = knowledge_augmentation(data_obj.raw_texts, data_obj.entity, strategy='back')
                        data_obj.x, _ = get_tf_idf_by_texts(texts, None, None, max_features=max_features, use_tokenizer=False)

                    elif typ == 'word2vec':
                        data_obj.x = get_word2vec(data_obj.raw_texts)
                    elif typ == 'bow':
                        if name.startswith("ptb"):
                            data_obj.x_mix = [[] for _ in range(5)]
                            for i in range(5):
                                print('len',len(data_obj.ptb_texts[i]))
                                ptb_idx = data_obj.mod_nodes[i]
                                ptb_text = data_obj.ptb_texts[i] + data_obj.raw_texts
                                ptb_feature = get_bow(ptb_text)
                                data_obj.x_mix[i] = ptb_feature[num_nodes:]
                                data_obj.ptb_features[i] = ptb_feature[:num_nodes]
                        else:
                            data_obj.x = get_bow(data_obj.raw_texts)
                    elif typ == 'sbert':
                        #if "know" not in name:
                        if name.startswith("ptb"):
                            for i in range(5):
                                print('len',len(data_obj.ptb_texts[i]))
                                ptb_idx = data_obj.mod_nodes[i]
                                ptb_text = data_obj.ptb_texts[i]
                                selected_texts = [ptb_text[idx] for idx in ptb_idx]
                                x = get_sbert_embedding(selected_texts)
                                new = torch.clone(data_obj.x)
                                new[ptb_idx] = x
                                data_obj.ptb_features[i] = new
                            # data_obj.x = get_sbert_embedding(data_obj.raw_texts)
                        else:
                            data_obj.x = get_sbert_embedding(data_obj.raw_texts)
                    elif typ == 'know_inp_sb':
                        texts_inp, _ = knowledge_augmentation(data_obj.raw_texts, data_obj.entity, strategy='inplace')
                        data_obj.x = get_e5_large_embedding(texts_inp, 'cuda', name + 'knowinp', batch_size=16)
                    elif typ == "know_sep_sb":
                        _, knowledge = knowledge_augmentation(data_obj.raw_texts, data_obj.entity, strategy='separate')
                        data_obj.x = get_e5_large_embedding(knowledge, 'cuda', name + 'knowsep', batch_size=16)
                    elif typ == 'ada':
                        if name in ['cora', 'citeseer', 'pubmed']:
                            data_obj.x = torch.tensor(openai_ada_api(data_obj.raw_texts))
                        elif name == 'arxiv':
                            data_obj.x = torch.load("./ogb_node_features.pt", map_location = 'cpu')
                        elif name == 'products':
                            with h5py.File('ogbn_products.h5', 'r') as hf:
                                numpy_array = np.array(hf['products'])
                                # convert the numpy array to a torch tensor
                                tensor = torch.from_numpy(numpy_array)
                                data_obj.x = tensor
                    elif typ == 'Anglellama':
                        print('llama')
                        if name == "pubmed" and setting == "random":
                            llama_obj = torch.load(osp.join(data_path, "new", "pubmed_fixed_llama.pt"), map_location="cpu")
                            data_obj.x = llama_obj.x
                        else:
                            print('---llama')
                            if name.startswith("ptb"):
                                for i in range(5):
                                    print('len', len(data_obj.ptb_texts[i]))
                                    ptb_idx = data_obj.mod_nodes[i]
                                    ptb_text = data_obj.ptb_texts[i]
                                    selected_texts = [ptb_text[idx] for idx in ptb_idx]
                                    x = get_Anglellama_embedding(selected_texts)
                                    new = torch.clone(data_obj.x)
                                    new[ptb_idx] = x
                                    data_obj.ptb_features[i] = new
                            else:
                                data_obj.x = get_Anglellama_embedding(data_obj.raw_texts)
                    elif typ == 'llama':
                        print('llama')
                        if name == "pubmed" and setting == "random":
                            llama_obj = torch.load(osp.join(data_path, "new", "pubmed_fixed_llama.pt"), map_location="cpu")
                            data_obj.x = llama_obj.x
                        else:
                            print('---llama')
                            data_obj.x = get_llama_embedding(data_obj.raw_texts)
                    elif typ == 'llm2vec':
                        data_obj.x = get_llm2vec_embedding(data_obj.raw_texts)
                    elif typ == 'llama_ft':
                        print('llama')
                        if name == "pubmed" and setting == "random":
                            llama_obj = torch.load(osp.join(data_path, "new", "pubmed_fixed_llama.pt"), map_location="cpu")
                            data_obj.x = llama_obj.x
                        else:
                            print('---llama')
                            data_obj.x = get_llama_ft_embedding(data_obj.raw_texts)
                    elif typ == "text-ada-embedding-002":
                        print('ada--')
                        data_obj.x = get_ada_embedding_002_embedding(data_obj.raw_texts)
                    elif typ == "palm-cortex-001":
                        data_obj.x = get_palm_cortex_001_embedding(data_obj.raw_texts)
                    elif typ == 'uae':
                        print('uae')
                        if name == "pubmed" and setting == "random":
                            llama_obj = torch.load(osp.join(data_path, "new", "pubmed_fixed_llama.pt"), map_location="cpu")
                            data_obj.x = llama_obj.x
                        else:
                            print('---uva')
                            data_obj.x = get_uae_embedding(data_obj.raw_texts)
                    elif typ == 'deberta':
                        print('deberta')
                        if name == "pubmed" and setting == "random":
                            llama_obj = torch.load(osp.join(data_path, "new", "pubmed_fixed_llama.pt"), map_location="cpu")
                            data_obj.x = llama_obj.x
                        else:
                            print('---deberta')
                            data_obj.x = get_deberta_embedding(data_obj.raw_texts)
                    elif typ == 'T5':
                        print('T5')
                        if name == "pubmed" and setting == "random":
                            llama_obj = torch.load(osp.join(data_path, "new", "pubmed_fixed_llama.pt"), map_location="cpu")
                            data_obj.x = llama_obj.x
                        else:
                            print('---T5')
                            data_obj.x = get_T5_embedding(data_obj.raw_texts)
                    elif typ == "ft":
                        if name == 'pubmed' or name == 'cora':
                            data_obj.xs = []
                            for i in range(5):
                                emb = np.memmap(f"./lmoutput/{name}_finetune_{setting}_{i}.emb", dtype=np.float16, mode='r',
                                    shape=(num_nodes, hidden_dim))
                                x = torch.tensor(emb, dtype=torch.float32)
                                data_obj.xs.append(x)
                            data_obj.x = data_obj.xs[0]
                        else:
                        # elif 'know' not in name:
                            emb = np.memmap(f"./lmoutput/{name}_finetune_{setting}_0.emb", dtype=np.float16, mode='r',
                                shape=(num_nodes, hidden_dim))
                            data_obj.x = torch.tensor(emb, dtype=torch.float32)
                    elif typ == "noft":
                        if name == 'pubmed' or name == 'cora':
                            data_obj.xs = []
                            for i in range(5):
                                emb = np.memmap(f"./lmoutput/{name}_no_finetune_{setting}_{i}.emb", dtype=np.float16, mode='r',
                                    shape=(num_nodes, hidden_dim))
                                x = torch.tensor(emb, dtype=torch.float32)
                                data_obj.xs.append(x)
                            data_obj.x = data_obj.xs[0]
                        else:
                            emb = np.memmap(f"./lmoutput/{name}_no_finetune_{setting}.emb", dtype=np.float16, mode='r',
                                shape=(num_nodes, hidden_dim))
                            data_obj.x = torch.tensor(emb, dtype=torch.float32)
                    elif typ == 'e5':
                        if name.startswith("ptb"):
                            for i in range(5):
                                print('len',len(data_obj.ptb_texts[i]))
                                ptb_idx = data_obj.mod_nodes[i]
                                ptb_text = data_obj.ptb_texts[i]
                                selected_texts = [ptb_text[idx] for idx in ptb_idx]
                                x = get_e5_large_embedding(selected_texts, 'cuda', name, batch_size=16)
                                emb = torch.load(f"./openai_out/{name}_e5_embedding.pt")
                                new = torch.clone(data_obj.x)
                                new[ptb_idx] = emb
                                data_obj.ptb_features[i] = new
                        else:
                            data_obj.x = get_e5_large_embedding(data_obj.raw_texts, 'cuda', name, batch_size=16)
                            emb = torch.load(f"./openai_out/{name}_e5_embedding.pt")
                            data_obj.x = emb
                    elif typ == 'google':
                        if name in ['arxiv', 'products']:
                            continue
                        emb = torch.load(f"./openai_out/{name}_google_embedding.pt")
                        emb = emb.reshape(num_nodes, -1)
                        data_obj.x = emb
                    elif typ == "know_exp_ft":
                        xs = []
                        for i in range(5):
                            emb = np.memmap(f"./lmoutput/{name}_finetune_{setting}_{i}_exp.emb", dtype=np.float16, mode='r',
                                shape=(num_nodes, hidden_dim))
                            x = torch.tensor(emb, dtype=torch.float32)
                            xs.append(x)
                        data_obj.xs = xs
                        data_obj.x = xs[0]
                    elif typ == "TAPE_ft_E":
                        LM_emb_path = f"./lmoutput/{name}-deberta-base-seed3.emb"
                        print(num_nodes)
                        print(f"LM_emb_path: {LM_emb_path}")
                        emb = np.memmap(f"./lmoutput/{name}-deberta-base-seed3.emb", dtype=np.float16, mode='r',
                                        shape=(num_nodes, hidden_dim))
                        x = torch.tensor(emb, dtype=torch.float32)
                        data_obj.x = x
                    elif typ == "know_inp_ft":
                        xs = []
                        for i in range(5):
                            emb = np.memmap(f"./lmoutput/{name}_inp_finetune_{setting}_{i}.emb", dtype=np.float16, mode='r',
                                shape=(num_nodes, hidden_dim))
                            x = torch.tensor(emb, dtype=torch.float32)
                            xs.append(x)
                        data_obj.xs = xs
                        data_obj.x = xs[0]
                    elif typ == "know_sep_ft":
                        xs = []
                        for i in range(5):
                            emb = np.memmap(f"./lmoutput/{name}_sep_finetune_{setting}_{i}.emb", dtype=np.float16, mode='r',
                                shape=(num_nodes, hidden_dim))
                            x = torch.tensor(emb, dtype=torch.float32)
                            xs.append(x)
                        data_obj.xs = xs
                        data_obj.x = xs[0]
                    elif typ == "know_exp_sb":
                        exp = torch.load(f"./preprocessed_data/new/{name}_explanation.pt")
                        data_obj.x = get_sbert_embedding(exp)
                    elif typ == "know_exp_e5":
                        exp = torch.load(f"./preprocessed_data/new/{name}_explanation.pt")
                        print(exp[2707])
                        # data_obj.x = get_sbert_embedding(exp)
                        data_obj.x = get_e5_large_embedding(exp, 'cuda', name + 'knowexp', batch_size=16)
                    elif typ == "pl":
                        pl = torch.load(f"./preprocessed_data/new/{name}_pred.pt")
                        data_obj.x = pl
                if name in ['cora', 'citeseer', 'pubmed']:
                    new_train_masks = []
                    new_val_masks = []
                    new_test_masks = []
                    for k in range(num_split := 10):
                        set_seed_config(split_seeds[k])
                        if setting == 'fixed':
                            ## 20 per class
                            fixed_split = LabelPerClassSplit(num_labels_per_class=20, num_valid = 500, num_test=1000)
                            t_mask, val_mask, te_mask = fixed_split(data_obj, data_obj.x.shape[0])
                            new_train_masks.append(t_mask)
                            new_val_masks.append(val_mask)
                            new_test_masks.append(te_mask)
                        else:
                            print('data_obj.x', data_obj.x)
                            total_num = data_obj.x.shape[0]
                            train_num = int(0.6 * total_num)
                            val_num = int(0.2 * total_num)
                            t_mask, val_mask, te_mask = generate_random_mask(data_obj.x.shape[0], train_num, val_num)
                            new_train_masks.append(t_mask)
                            new_val_masks.append(val_mask)
                            new_test_masks.append(te_mask)
                    data_obj.train_masks = new_train_masks
                    data_obj.val_masks = new_val_masks
                    data_obj.test_masks = new_test_masks


                torch.save(data_obj, osp.join(data_path, "new", f"{name}_{setting}_{typ}.pt"))
                print("Save object {}".format(osp.join(data_path, "new", f"{name}_{setting}_{typ}.pt")))







if __name__ == '__main__':
    set_api_key()
    main()