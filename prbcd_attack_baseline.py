from deeprobust.graph.utils import *
from args import *
from data import get_dataset, set_seed_config, set_api_key, pkl_and_write, get_tf_idf_by_texts
import torch
from train_utils import train, test, test1, get_optimizer, confidence_test, topk_test, to_inductive, batch_train, batch_test
from deeprobust.graph.defense_pyg import GCN
import numpy as np
import ipdb
import optuna
from torch.utils.tensorboard import SummaryWriter
import openai
from copy import deepcopy
import logging
import time
from torch_geometric.utils import index_to_mask
import optuna
import sys
from hyper import hyper_search
import os.path as osp
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from utils import delete_non_tensor_attributes
from ogb.nodeproppred import Evaluator
from collections import defaultdict
from deeprobust.graph.global_attack import MetaApprox, Metattack,PRBCD,PGDAttack
from deeprobust.graph.utils import *
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(15)
torch.manual_seed(15)
if device != 'cpu':
    torch.cuda.manual_seed(15)

def sweep(args = None):

    sweep_seeds = [0, 1, 2, 3, 4]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"{args.dataset}_{args.model_name}_{args.data_format}_{args.split}"
    study = optuna.create_study(study_name=study_name, storage=None, direction='maximize', load_if_exists=True)
    param_f = hyper_search
    sweep_round = args.sweep_round
    study.optimize(lambda trial: sweep_run(trial, args, sweep_seeds, param_f, device), catch=(RuntimeError,), n_trials=sweep_round, callbacks=[lambda study, trial: max_trial_callback(study, trial, sweep_round)], show_progress_bar=True, gc_after_trial=True)
    main(args=args, custom_args = study.best_trial.params, save_best = True)
    print(study.best_trial.params)
def test(data, new_adj, edge_weight, train_mask, val_mask, test_mask, agent, gcn=None):
    ''' test on GCN '''

    if gcn is None:
        data = data.to(device)
        data.y = data.y.cpu()
        gcn = GCN(nfeat=args.input_dim,
              nhid=args.hidden,
              nclass=args.num_classes,
              dropout=args.dropout, device=device, lr=args.lr)
        gcn = gcn.to(device)
        # gcn.fit(features, new_adj, labels, idx_train) # train without model picking
        gcn.fit(data.x, new_adj, data.y, train_mask, val_mask, patience=30) # train with validation model picking
        gcn.eval()
        output, logits = agent.model.predict(data.x, new_adj, edge_weight)
        loss_test, acc_test = agent.get_perf(output, data.y, test_mask)

        del gcn

    else:
        gcn.eval()
        data = data.to(device)
        output, logits = agent.model.predict(data.x, new_adj, edge_weight)
        loss_test, acc_test = agent.get_perf(output, data.y, test_mask)
        del gcn

    return acc_test, logits


def main(args = None, custom_args = None, save_best = False):
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if custom_args != None:
        args = replace_args_with_dict_values(args, custom_args)
    if not args.batchify and args.ensemble_string == "":
        data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test)
    elif args.ensemble_string != "":
        data = get_dataset(args.seed_num, args.dataset, args.split, 'sbert', args.low_label_test)


    data.x = data.x.to(torch.float32)
    seeds = [i for i in range(args.seed_num)]
    epoch = args.epochs
    vars(args)['input_dim'] = data.x.shape[1]
    vars(args)['num_classes'] = data.y.max().item() + 1


    if args.model_name == 'LP':
        need_train = False
    else:
        need_train = True
    #############################attck

    data.y = data.y.cpu()
    clean_sum = []
    eva_sum = []
    poi_sum = []
    execute_attack = True
    attack_adj = None
    execute_attack = True
    if not args.batchify and args.ensemble_string == "":
        for i in range(5):

            data.train_mask = data.train_masks[i].cpu()
            data.val_mask = data.val_masks[i].cpu()
            data.test_mask = data.test_masks[i].cpu()
            data.train_mask = torch.where(data.train_mask == True)[0].numpy()
            data.val_val = torch.where(data.val_mask == True)[0].numpy()
            data.test_mask = torch.where(data.test_mask == True)[0].numpy()
            print('train_mask', data.train_mask)

            if args.dataset == "arxiv":

                nclass = max(data.y).item() + 1
                print('nclass', nclass)
                model = GCN(nfeat=data.x.shape[1], nhid=256, dropout=0,
                            nlayers=3, with_bn=True, weight_decay=1e-4, nclass=nclass,lr=0.001,
                            device=device).to(device)
                print(model)
                agent = PRBCD(data, model=model, device=device, epochs=100)  #100 by default, we are attacking the GCN model
                model, clean_test, logits = agent.pretrain_model(model)  # use the function to pretrain the provided model
                clean_sum.append(clean_test)

            if args.dataset == "history":
                nclass = max(data.y).item() + 1
                print('nclass', nclass)
                model = GCN(nfeat=data.x.shape[1], nhid=256, nclass=nclass,
                            nlayers=2, dropout=0.5, lr=0.001, weight_decay=1e-4,
                            device=device).to(device)
                agent = PRBCD(data, model=model, device=device, epochs=50)  # by default, we are attacking the GCN model
                model, clean_test, logits = agent.pretrain_model(model)  # use the function to pretrain the provided model
                clean_sum.append(clean_test)
            if args.ptb_rate != 0:
                if execute_attack:
                    edge_index, edge_weight, acc_test = agent.attack(ptb_rate=args.ptb_rate)
                modified_adj = edge_index
                modified_edge_weight = edge_weight
                print('=== testing GCN on Evasion attack ===')

                evasion_acc, logits = test(data, modified_adj, modified_edge_weight, data.train_mask, data.val_mask, data.test_mask,agent, agent.model)
                eva_sum.append(evasion_acc)


            del agent
        print('=== testing GCN on clean graph ===')
        clean_mean_test_acc = np.mean(clean_sum) * 100
        clean_std_test_acc = np.std(clean_sum) * 100
        print(f"clean Test Accuracy: {clean_mean_test_acc:.2f} ± {clean_std_test_acc:.2f}")
        print("clean Test acc: {}".format(clean_sum))
        if args.ptb_rate != 0:
            eva_mean_test_acc = np.mean(eva_sum) * 100
            eva_std_test_acc = np.std(eva_sum) * 100
            print(f"attack Test Accuracy: {eva_mean_test_acc:.2f} ± {eva_std_test_acc:.2f}")
            print("attack Test acc: {}".format(eva_sum))

    elif args.ensemble_string != "":
        feats = args.ensemble_string.split(";")
        res = []
        sep_test_acc = defaultdict(list)
        labels = data.y
        test_masks = data.test_masks
        type_res=[]
        type_res_evasion=[]
        sep_test_acc_evasion = defaultdict(list)

        for feat in feats:
            vars(args)['data_format'] = feat
            data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test)
            res=[]
            res_evasion = []
            res_poison = []
            clean_sum_ensemble = []
            evasion_sum_ensemble = []
            poison_sum_ensemble = []
            for i in range(5):
                data.train_mask = data.train_masks[i].cpu()
                data.val_mask = data.val_masks[i].cpu()
                data.test_mask = data.test_masks[i].cpu()
                data.train_mask = torch.where(data.train_mask == True)[0].numpy()
                data.val_val = torch.where(data.val_mask == True)[0].numpy()
                data.test_mask = torch.where(data.test_mask == True)[0].numpy()

                if args.dataset == "arxiv":
                    nclass = max(data.y).item() + 1
                    print('nclass', nclass)
                    model = GCN(nfeat=data.x.shape[1], nhid=256, dropout=0,
                                nlayers=3, with_bn=True, weight_decay=1e-4, nclass=nclass, lr=0.001,
                                device=device).to(device)
                    print(model)
                    agent = PRBCD(data, model=model, device=device,
                                  epochs=100)  # 100 by default, we are attacking the GCN model
                    model, clean_test, logits = agent.pretrain_model(
                        model)  # use the function to pretrain the provided model

                if args.ptb_rate == 0:
                    clean_sum_ensemble.append(clean_test)
                    res.append(logits.detach().cpu())
                    del logits


                if args.ptb_rate != 0:
                    if execute_attack:
                        edge_index, edge_weight, acc_test = agent.attack(ptb_rate=args.ptb_rate)
                    modified_adj = edge_index
                    modified_edge_weight = edge_weight
                    print('=== testing GCN on Evasion attack ===')

                    evasion_acc, evasion_logits = test(data, modified_adj, modified_edge_weight, data.train_mask, data.val_mask,
                                               data.test_mask, agent, agent.model)
                    evasion_sum_ensemble.append(evasion_acc)
                    res_evasion.append(evasion_logits.detach().cpu())
                    del evasion_logits
            if args.ptb_rate == 0:
                type_res.append(res)
                sep_test_acc[feat] = clean_sum_ensemble
                print('clean', sep_test_acc)
            else:
                type_res_evasion.append(res_evasion)
                sep_test_acc_evasion[feat] = evasion_sum_ensemble

                del res_evasion
        if args.ptb_rate == 0:
            print('==========clean-ensemble==========')
            for key, value in sep_test_acc.items():
                print('value',value)
                mean = np.mean(value) * 100
                std = np.std(value) * 100
                print(f"{key}: {mean:.2f} ± {std:.2f}")
            ensemble_input = [[type_res[i][j] for i in range(len(feats))] for j in range(len(seeds))]
            ensemble_helper(ensemble_input, data.y, test_masks)
        else:
            print('==========evasion==========')
            for key, value in sep_test_acc_evasion.items():
                mean = np.mean(value) * 100
                std = np.std(value) * 100
                print(f"{key}: {mean:.2f} ± {std:.2f}")
            ensemble_input = [[type_res_evasion[i][j] for i in range(len(feats))] for j in range(len(seeds))]
            ensemble_helper(ensemble_input, data.y, test_masks)
            del type_res_evasion

        torch.cuda.empty_cache()

@torch.no_grad()
def ensemble_helper(logits, labels, test_masks):
    seeds_num = len(logits)
    accs = []
    for i in range(seeds_num):
        test_mask = test_masks[i]
        this_seed_logits = logits[i]
        avg_logits = sum(this_seed_logits) / len(this_seed_logits)
        pred = torch.argmax(avg_logits, dim=1).cpu()
        labels = labels.cpu()
        acc = torch.sum(pred[test_mask] == labels[test_mask]).item() / len(labels[test_mask])
        accs.append(acc)
    mean_test_acc = np.mean(accs) * 100.0
    std_test_acc = np.std(accs) * 100.0
    print(f"Ensemble Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
    return mean_test_acc, std_test_acc

if __name__ == '__main__':
    current_time = int(time.time())
    print('start')
    logging.basicConfig(filename='./logs/{}.log'.format(current_time),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

    args = get_command_line_args()
    set_api_key()

    if args.mode == "main":
        main(args = args)
    else:
        sweep(args = args)

