import argparse
import torch
from attacks.pyg_defence import GCN, SAGE
from deeprobust.graph.utils import *
from copy import deepcopy
import json

from attacks.models import *
import attacks.models as models
from attacks.node_prbcd_v1 import *
from raw_data.data import *
import pickle
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--edge_ptb_rate', type=float, default=0, help='edge perturbation rate.')
parser.add_argument('--node_ptb_rate', type=float, default=0.05, help='node perturbation rate.')
parser.add_argument('--sample_choice', type=str, default='random', choices=["random", "large", "small", "cluster"])
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--const', type=int, default=1e6)
parser.add_argument('--seed', type=int, default=15)
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'pubmed', 'arxiv', 'citeseer2', 'history', 'wikics'], help='dataset')
parser.add_argument('--data_type', type=str, default='fixed', choices=['fixed', 'ig', 'finetuned'])
parser.add_argument('--model', type=str, default='sbert', choices=['sbert', 'e5', 'llama', 'bow', 'llama_citeseer'], help='embedding model')
parser.add_argument('--finetune', type=bool, default=False)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--test_ptb', type=bool, default=False)
args = parser.parse_args()

def test(data_obj, clean_out = None, modified_idx = None, gcn=None, ptb_output = None):
    ''' test on GCN '''
    gcn.eval()
    
    output = gcn.forward(x = data_obj.x.to(device), edge_index = data_obj.edge_index.to(device), edge_weight = data_obj.edge_weight)
    output = output.detach().cpu()
    # output = text_encoder.mlp(data_obj.x)
    print(len(data_obj.test_mask))
    loss_test = F.nll_loss(output[data_obj.test_mask], data_obj.y[data_obj.test_mask].cpu())
    acc_test = accuracy(output[data_obj.test_mask], data_obj.y[data_obj.test_mask].cpu())
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    if modified_idx is not None:
        if ptb_output is not None:
            mod_out = ptb_output
        else:
            mod_out = output[modified_idx]
        
        mod_label = data_obj.y[modified_idx].cpu()
        clean = clean_out[modified_idx]
        
        # print(f"mod_label: {mod_label}, mod_predict: {clean.argmax(1)}, mod_mod_predict: {mod_out.argmax(1)}")

        mod_acc_ptb = accuracy(mod_out, mod_label)
        mod_acc_ori = accuracy(clean, mod_label)
        
        ori_preds = clean.argmax(1)
        mod_preds = mod_out.argmax(1)
        cor_idx = torch.where(ori_preds == mod_label)[0]
        # print(len(cor_idx))
        s_attacked= torch.sum(ori_preds[cor_idx] != mod_preds[cor_idx])/len(cor_idx)

        return acc_test.item(), mod_acc_ori, mod_acc_ptb, s_attacked

    else:
        return acc_test.item(), 0, 0, 0


def run(data, split):

    # data.x = torch.FloatTensor(normalize_feature(data.x.cpu())).to(device)

    features = data.x
    labels = data.y

    print('Attacking model...')
    if args.dataset == "arxiv":
        dropout = 0.0
        nlayers = 3
        lr = 0.001
        bn = True
        decay = 1e-4
        if args.model == 'llama' or args.model == 'bow':
            mod_nodes = data.mod_nodes
        else:
            mod_nodes = None
        
    else:
        dropout = 0.5
        nlayers = args.nlayers
        lr = 0.01
        bn = False
        decay = 5e-4
        mod_nodes = None
    
    if args.node_ptb_rate != 0:
        epochs = 100
    else:
        epochs = 100

    model = GCN(nfeat=features.shape[1], nhid=256, dropout=dropout,
                    nlayers=nlayers, with_bn=bn, weight_decay=decay, nclass=labels.max().item() + 1, lr = lr,
                    device=device).to(device)
    # 
    # model = SAGE(nfeat=features.shape[1], nhid=32, dropout=dropout,
    #              nlayers=nlayers, with_bn=False, weight_decay=5e-4, nclass=labels.max().item() + 1,
    #              device=device).to(device)
    # model = new_MLP(nfeat=features.shape[1], nhid=256, dropout=dropout,
    #                 nlayers=2, with_bn=bn, weight_decay=5e-4, nclass=labels.max().item() + 1,
    #                 device=device).to(device)

    ptb_nnodes = int(data.x.shape[0]*args.node_ptb_rate)
    ptb_nnodes = 200
    print(f"number of nodes to perturb: {ptb_nnodes}")
    print("##", text_encoder.embedding.weight.shape[1])
    agent = NodePRBCD(model=model, data = data, text_encoder= text_encoder, sample_choice = args.sample_choice, mod_nodes = mod_nodes, word_embed_size = text_encoder.embedding.weight.shape[1], device = device, node_search_space_size = ptb_nnodes, epochs = epochs, const_scale = args.const)

    model, clean_acc, clean_output = agent.pretrain_model(model) # use the function to pretrain the provided model

    evasion_acc = 0

    if args.edge_ptb_rate == 0 and args.node_ptb_rate == 0:
        print("==== clean graph ====")
        clean_acc, mod_acc_ori, mod_acc_ptb, s_attacked = test(data, clean_out = clean_output, modified_idx = None, gcn=model)
        if args.test_ptb:
            tmp_data = deepcopy(data)
            tmp_data.x = tmp_data.ptb_features[split]
            # tmp_data.x = torch.FloatTensor(normalize_feature(tmp_data.x.cpu())).to(device)
            # tmp_data.edge_index = tmp_data.ptb_edge_index[split]
            # tmp_data.edge_weight = tmp_data.ptb_edge_weight[split]
            evasion_acc, mod_acc_ori, mod_acc_ptb, s_attacked = test(tmp_data, clean_out = clean_output, modified_idx = tmp_data.mod_nodes[split], gcn=model)

    else:
        print("==== Start Attacking ====")
        mod_feat, mod_nodes, perturbed_seqs, mod_edge_index, edge_weight = agent.attack(edge_ptb_rate=args.edge_ptb_rate)
        print("==== evasion attack ====")
        data.edge_index = mod_edge_index
        data.x = mod_feat
        data.features = mod_feat
        data.perturbed_seqs = perturbed_seqs
        data.mod_nodes = mod_nodes
        data.edge_weight = edge_weight

        if args.node_ptb_rate != 0:
            seq_lens = []
            for seq in perturbed_seqs:
                seq_lens.append(len(torch.where(seq!=0)[0]))
            seq_lens = torch.LongTensor(seq_lens)

            new_feats = torch.LongTensor([]).to(device)
            for idx, seq in enumerate(perturbed_seqs):
                feat = text_encoder.generate(input_ids = seq.unsqueeze(0), seq_len = seq_lens[idx].unsqueeze(0))
                new_feats = torch.cat([new_feats, feat])
            data.x[mod_nodes] = new_feats.cpu().float()
            # print("feature difference: ", torch.sum(new_feats != mod_feat[mod_nodes]))

        # new_output = text_encoder(input_ids = perturbed_seqs, seq_len = seq_lens)['pred']
        # fake_weight = torch.zeros_like(data.edge_index[0]).to(torch.float32)
        # new_output = model.forward(x = data.x, edge_index = data.edge_index, edge_weight = data.edge_weight)
        # new_output = new_output[mod_nodes]

        evasion_acc, mod_acc_ori, mod_acc_ptb, s_attacked = test(data, clean_out = clean_output, modified_idx = mod_nodes, gcn=model, ptb_output = None)
        if args.save:
            torch.save(data, f'ig_results/{args.dataset}_{args.model}/edge_{args.edge_ptb_rate}_node_{args.node_ptb_rate}_{args.sample_choice}/split_{split}.pt')
            # torch.save(mod_edge_index, f"arxiv_mod_adj_0.05.pt")
    return data, clean_acc, evasion_acc, mod_acc_ori, mod_acc_ptb, s_attacked

def eval_seqs(ptb_seqs, ori_seqs, ptb_nodes):
    diff = 0.0
    avg_len = 0.0
    for i, idx in enumerate(ptb_nodes):
        ptb_seq = ptb_seqs[i]
        ori_seq = torch.LongTensor(ori_seqs[idx])
        avg_len += len(ori_seq)
        for j, token in enumerate(ori_seq):
            if ptb_seq[j].item() != token.item():
                diff += 1

    return avg_len/len(ori_seqs), diff/len(ori_seqs)
    
if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    if args.dataset != 'arxiv':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True)
    if not os.path.exists(f'ig_results/{args.dataset}_{args.model}'):
        os.mkdir(f'ig_results/{args.dataset}_{args.model}')
    if not os.path.exists(f'ig_results/{args.dataset}_{args.model}/edge_{args.edge_ptb_rate}_node_{args.node_ptb_rate}_{args.sample_choice}'):
        os.mkdir(f'ig_results/{args.dataset}_{args.model}/edge_{args.edge_ptb_rate}_node_{args.node_ptb_rate}_{args.sample_choice}')
    set_seed_config(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.finetune:
        ft_data = args.dataset
    else:
        ft_data = None

    # path = f"processed_data/final_processed_{args.dataset}.pt"
    if args.dataset != "arxiv" or (args.model != 'bow' and args.model != 'llama'):
        path = f"raw_data/{args.dataset}_{args.data_type}_{args.model}.pt"
        # dataset = torch.load(f"processed_data/enhanced/ptbciteseer2_fixed_sbert.pt")
        print(path)
        dataset = torch.load(path)
        dataset.x = torch.FloatTensor(dataset.x)
        # dataset.y = torch.FloatTensor(dataset.y)
        # dataset = torch.load("perturbed_cora_0.05_sbert.pt")
        text_encoder = models.BertC(model_name = args.model, dropout=0.5, num_class=7, ft_data = ft_data, device = device)
    
    print(f"{args.dataset} initiated!")

    clean_result = []
    evasion_result = []
    mod_acc_ori_result = []
    mod_acc_ptb_result = []
    s_attacked_result = []
    ori_seq_len_result = []
    diff_rate_result = []

    num_splits = 1

    for split in range(num_splits):
        print(f"running split: {split}")
        if args.dataset == "arxiv" and (args.model == 'bow' or args.model == 'llama'):
            data = torch.load(f"raw_data/arxiv_fixed_{args.model}_{split}.pt")
            print("arxiv loaded")
        else:
            data = deepcopy(dataset)
        #
        data.train_mask = data.train_masks[split].cpu()
        data.val_mask = data.val_masks[split].cpu()
        data.test_mask = data.test_masks[split].cpu()
        data.train_mask = torch.where(data.train_mask == True)[0].numpy()
        data.val_mask = torch.where(data.val_mask == True)[0].numpy()
        data.test_mask = torch.where(data.test_mask == True)[0].numpy()
        data.y = torch.LongTensor(data.y)

        # # Sbert
        if args.model == 'bow' or args.model == 'tfidf':
            text_encoder = models.BOW(data, 2048, split, device)
            # data.x = data.clean_x[split]
            # data.seqs = data.clean_seqs[split]
            # data.candidates = data.clean_candidates[split]

        # # e5
        # text_encoder = models.e5(dropout=0.5, num_class=7, device = device).to(device)
        # # llama
        # text_encoder = models.llama(dropout=0.5, num_class=7, device = device).to(device)

        text_encoder.eval()

        # data.x = text_encoder.generate(txt = data.raw_texts)

        # run 
        data, clean_acc, evasion_acc, mod_acc_ori, mod_acc_ptb, s_attacked = run(data, split)
        # eval 
        if args.node_ptb_rate != 0:
            # ori_seq_len, diff_rate = eval_seqs(data.perturbed_seqs, data.seqs, data.mod_nodes)
            ori_seq_len = 0 
            diff_rate = 0
        else:
            ori_seq_len = 0 
            diff_rate = 0

        clean_result.append(clean_acc)
        evasion_result.append(evasion_acc)
        mod_acc_ori_result.append(mod_acc_ori)
        mod_acc_ptb_result.append(mod_acc_ptb)
        s_attacked_result.append(s_attacked)
        ori_seq_len_result.append(ori_seq_len)
        diff_rate_result.append(diff_rate)
        del data

    clean_result = torch.Tensor(clean_result)
    evasion_result = torch.Tensor(evasion_result)
    mod_acc_ori_result = torch.Tensor(mod_acc_ori_result)
    mod_acc_ptb_result = torch.Tensor(mod_acc_ptb_result)
    s_attacked_result = torch.Tensor(s_attacked_result)
    avg_seq_len_result = torch.Tensor(ori_seq_len_result)
    diff_rate_result = torch.Tensor(diff_rate_result)

    print(f"""
            clean: {clean_result.mean()} ± {clean_result.std()},
            evasion attack: {evasion_result.mean()} ± {evasion_result.std()},
            mod_acc_ori: {mod_acc_ori_result.mean()} ± {mod_acc_ori_result.std()},
            mod_acc_ptb: {mod_acc_ptb_result.mean()} ± {mod_acc_ptb_result.std()},
            s_attacked: {s_attacked_result.mean()} ± {s_attacked_result.std()},
            avg_seq_len: {avg_seq_len_result.mean()} ± {avg_seq_len_result.std()},
            diff_rate: {diff_rate_result.mean()} ± {diff_rate_result.std()},
            """)
    result = {"clean": [float(clean_result.mean().item()), float(clean_result.std().item())], 
              "evasion": [float(evasion_result.mean().item()), float(evasion_result.std().item())],
              "mod_acc_ori": [float(mod_acc_ori_result.mean().item()), float(mod_acc_ori_result.std().item())],
              "mod_acc_ptb": [float(mod_acc_ptb_result.mean().item()), float(mod_acc_ptb_result.std().item())],
              "s_attacked": [float(s_attacked_result.mean().item()), float(s_attacked_result.std().item())],
              "avg_seq_len": [float(avg_seq_len_result.mean().item()), float(avg_seq_len_result.std().item())],
              "diff_rate": [float(diff_rate_result.mean().item()), float(diff_rate_result.std().item())]
              }
    if args.save:
        with open(f'ig_results/{args.dataset}_{args.model}/edge_{args.edge_ptb_rate}_node_{args.node_ptb_rate}_{args.sample_choice}.json','w') as f:
            json.dump(result,f) 
