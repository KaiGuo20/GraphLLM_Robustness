import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from torch_geometric.nn.models import MLP
import argparse
import os

from sklearn.feature_extraction.text import TfidfVectorizer

from args import *
from data import get_dataset, set_seed_config, set_api_key, pkl_and_write, get_tf_idf_by_texts
import torch
from train_utils import train, test, test1, get_optimizer, confidence_test, topk_test, to_inductive, batch_train, batch_test
from models import get_model
from models import GCN as GCN1
from deeprobust.graph.defense_pyg import GCN as pyg_GCN
from deeprobust.graph.defense import GCN
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
from torch_sparse import SparseTensor
from torch_geometric.utils import homophily
from models1 import new_MLP

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
def test(data, new_adj, train_mask, val_mask, test_mask, gcn=None):
    ''' test on GCN '''

    if gcn is None:
        data = data.to(device)
        print('gcn is None')
        data.y = data.y.cpu()
        gcn = GCN(nfeat=args.input_dim,
              nhid=args.hidden,
              nclass=args.num_classes,
              dropout=args.dropout, device=device, lr=args.lr)
        gcn = gcn.to(device)
        # gcn.fit(features, new_adj, labels, idx_train) # train without model picking
        gcn.fit(data.x, new_adj, data.y, train_mask, val_mask, patience=30) # train with validation model picking
        gcn.eval()
        output, logits = gcn.predict()
        loss_test = F.nll_loss(output[test_mask].cpu(), data.y[test_mask].cpu())
        acc_test = accuracy(output[test_mask], data.y[test_mask])

    else:

        data = data.to(device)
        gcn.eval()
        output, logits = gcn.predict(data.x.to(device), new_adj.to(device))
        loss_test = F.nll_loss(output[test_mask].cpu(), data.y[test_mask].cpu())
        acc_test = accuracy(output[test_mask], data.y[test_mask])

    return acc_test.item(), output

def main(args = None, custom_args = None, save_best = False):
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    execute_attack = True
    attack_adj = None
    if not args.batchify and args.ensemble_string == "":
        if custom_args != None:
            args = replace_args_with_dict_values(args, custom_args)
        if not args.batchify and args.ensemble_string == "":
            data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test)
        elif args.ensemble_string != "":
            data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test)

        vars(args)['input_dim'] = data.x.shape[1]
        vars(args)['num_classes'] = data.y.max().item() + 1

        if args.model_name == 'LP':
            need_train = False
        else:
            need_train = True
        #############################attck
        if args.attack != 'None':
            #########################
            import numpy as np
            from scipy.sparse import coo_matrix
            from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
            from scipy.sparse import csr_matrix
            print(data.edge_index.device)
            data.x = data.x.numpy()
            data.y = data.y.numpy()
            if args.dataset == 'arxiv':
                data.edge = torch.load('./arxiv_edge.pt')
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).toarray()
                adj[np.where(adj > 1)] = 1
                data.edge_index = from_scipy_sparse_matrix(coo_matrix(adj))[0].cpu()
                data.edge = csr_matrix((to_scipy_sparse_matrix(data.edge_index).toarray()))
            #########################################
            print('---------')
            perturbations = int(args.ptb_rate * (data.edge.sum() // 2))
            print(perturbations)
            adj, features, labels = data.edge, data.x, data.y
            print(features)
            if args.data_format != 'sbert' and args.data_format != 'llama_ft' and args.data_format != 'TAPE_ft_E' and args.data_format != 'llama'and args.data_format != 'Anglellama':
                print('normalize--')
                data.x = normalize_feature(features)
            data.edge, data.x, data.y = preprocess(adj, data.x, labels, preprocess_adj=False)
            if args.dataset == 'wikics':
                diagonal_elements = torch.diagonal(data.edge)
                data.edge = data.edge - torch.diag(diagonal_elements)

        data = data.to(device)
        # data.y = data.y.cpu()
        clean_sum = []
        text_sum = []
        eva_sum = []
        poi_sum = []
        attack_success = []
        for i in range(5):

            data.train_mask = data.train_masks[i].cpu()
            data.val_mask = data.val_masks[i].cpu()
            data.test_mask = data.test_masks[i].cpu()
            data.train_mask = torch.where(data.train_mask == True)[0].numpy()
            data.val_val = torch.where(data.val_mask == True)[0].numpy()
            data.test_mask = torch.where(data.test_mask == True)[0].numpy()

            target_gcn = GCN(nfeat=args.input_dim,
                      nhid=args.hidden,
                      nclass=args.num_classes,
                      dropout=args.dropout, device=device, lr=args.lr)
            target_gcn = target_gcn.to(device)
            target_gcn.fit(data.x, data.edge, data.y, data.train_mask, data.val_mask, patience=30)
            if args.ptb_rate == 0:
                print('=== testing GCN on clean graph ===')
                clean_acc, logits = test(data, data.edge, data.train_mask, data.val_mask, data.test_mask, target_gcn)
                print('clean acc', clean_acc)
                clean_sum.append(clean_acc)

            else:
                print('=== testing GCN on clean graph ===')
                clean_acc, logits = test(data, data.edge, data.train_mask, data.val_mask, data.test_mask, target_gcn)
                print('clean acc', clean_acc)
                clean_sum.append(clean_acc)
            # Setup Attack Model
                if execute_attack:
                    print('=== setup attack model ===')
                    model = PGDAttack(model=target_gcn, nnodes=data.edge.shape[0], loss_type='CE', device=device)
                    model = model.to(device)
                    fake_labels, logits = target_gcn.predict(data.x.to(device), data.edge.to(device))
                    fake_labels = torch.argmax(fake_labels, 1).cpu()
                    idx_fake = np.concatenate([data.train_mask, data.test_mask])
                    print('data.train_masks', data.train_masks)
                    fake_labels[data.train_mask] = data.y[data.train_mask].cpu()
                    model.attack(data.x.cpu(), data.edge.cpu(), fake_labels, idx_fake, perturbations, epochs=100)
                    attack_adj = model.modified_adj

                print('=== testing GCN on Evasion attack ===')
                edge_indices = torch.nonzero(attack_adj, as_tuple=False).t()
                print('data.attack_adj', edge_indices)
                modified_adj = attack_adj
                evasion_acc, logits = test(data, modified_adj, data.train_mask, data.val_mask, data.test_mask,target_gcn)
                eva_sum.append(evasion_acc)


                print('=== testing GCN on Poisoning attack ===')
                modified_adj = attack_adj
                poison_acc, logits = test(data, modified_adj, data.train_mask, data.val_mask, data.test_mask,)
                poi_sum.append(poison_acc)
        if args.ptb_rate == 0:
            clean_mean_test_acc = np.mean(clean_sum) * 100
            clean_std_test_acc = np.std(clean_sum) * 100
            print(f"clean Test Accuracy: {clean_mean_test_acc:.2f} ± {clean_std_test_acc:.2f}")
            print("clean Test acc: {}".format(clean_sum))

        else:
            eva_mean_test_acc = np.mean(eva_sum) * 100
            eva_std_test_acc = np.std(eva_sum) * 100
            print(f"Evasion Test Accuracy: {eva_mean_test_acc:.2f} ± {eva_std_test_acc:.2f}")
            print("Evasion Test acc: {}".format(eva_sum))

            poi_mean_test_acc = np.mean(poi_sum) * 100
            poi_std_test_acc = np.std(poi_sum) * 100
            print(f"Poisoning Test Accuracy: {poi_mean_test_acc:.2f} ± {poi_std_test_acc:.2f}")
            print("Poisoning Test acc: {}".format(poi_sum))

    elif args.ensemble_string != "":
        feats = args.ensemble_string.split(";")
        res = []
        sep_test_acc = defaultdict(list)

        first_dataX = None
        type_res=[]
        type_res_evasion=[]
        sep_test_acc_evasion = defaultdict(list)
        type_res_poison=[]
        sep_test_acc_poison = defaultdict(list)
        for feat in feats:
            vars(args)['data_format'] = feat
            data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test)
            seeds = [i for i in range(args.seed_num)]
            labels = data.y
            test_masks = data.test_masks
            print('data.x', data.x.size())
            vars(args)['input_dim'] = data.x.shape[1]
            vars(args)['num_classes'] = data.y.max().item() + 1
            if args.attack != 'None':
                #########################
                import numpy as np
                from scipy.sparse import coo_matrix
                from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
                from scipy.sparse import csr_matrix
                print(data.edge_index)
                data.x = data.x.numpy()
                data.y = data.y.numpy()
                if args.dataset == 'arxiv':
                    data.edge = torch.load('./arxiv_edge.pt')
                else:
                    adj = to_scipy_sparse_matrix(data.edge_index).toarray()
                    adj[np.where(adj>1)] = 1
                    data.edge_index = from_scipy_sparse_matrix(coo_matrix(adj))[0].cpu()
                    data.edge = csr_matrix((to_scipy_sparse_matrix(data.edge_index).toarray()))
                #########################################
                perturbations = int(args.ptb_rate * (data.edge.sum() // 2))
                print(perturbations)
                adj, features, labels = data.edge, data.x, data.y
                if args.data_format != 'sbert' and args.data_format != 'llama_ft' and args.data_format != 'llama'and args.data_format != 'Anglellama':
                    print('normalize--')
                    data.x = normalize_feature(features)
                data.edge, data.x, data.y = preprocess(adj, data.x, labels, preprocess_adj=False)
                print('data.edge', data.edge.size())

            if first_dataX is None:
                first_dataX = data.x
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
                target_gcn = GCN(nfeat=args.input_dim,
                                 nhid=args.hidden,
                                 nclass=args.num_classes,
                                 dropout=args.dropout, device=device, lr=args.lr)

                target_gcn = target_gcn.to(device)
                target_gcn.fit(data.x, data.edge, data.y, data.train_mask, data.val_mask, patience=30)
                if args.ptb_rate == 0:
                    print('=== testing GCN on clean graph ===')
                    clean_acc, logits = test(data, data.edge, data.train_mask, data.val_mask, data.test_mask, target_gcn)
                    print('clean acc', clean_acc)
                    clean_sum_ensemble.append(clean_acc)
                    res.append(logits)


                else:
                    print('=== testing GCN on clean graph ===')
                    clean_acc, logits = test(data, data.edge, data.train_mask, data.val_mask, data.test_mask, target_gcn)
                    print('clean acc', clean_acc)
                    clean_sum_ensemble.append(clean_acc)
                    res.append(logits)
                    # Setup Attack Model
                    if execute_attack:
                        print('=== setup attack model ===')
                        # data.x = first_dataX
                        model = PGDAttack(model=target_gcn, nnodes=data.edge.shape[0], loss_type='CE', device=device)
                        model = model.to(device)
                        fake_labels, logits = target_gcn.predict(data.x.to(device), data.edge.to(device))
                        fake_labels = torch.argmax(fake_labels, 1).cpu()
                        idx_fake = np.concatenate([data.train_mask, data.test_mask])
                        print('data.train_masks', data.train_masks)
                        fake_labels[data.train_mask] = data.y[data.train_mask].cpu()
                        model.attack(data.x.cpu(), data.edge.cpu(), fake_labels, idx_fake, perturbations, epochs=100)
                        attack_adj = model.modified_adj
                        # execute_attack = False
                        del model
                    print('=== testing GCN on Evasion attack ===')

                    modified_adj = attack_adj
                    evasion_acc, evasion_logits = test(data, modified_adj, data.train_mask, data.val_mask, data.test_mask, target_gcn)
                    evasion_sum_ensemble.append(evasion_acc)
                    res_evasion.append(evasion_logits)

                    # modified_features = model.modified_features
                    print('=== testing GCN on Poisoning attack ===')
                    modified_adj = attack_adj
                    poison_acc, poison_logits = test(data, modified_adj, data.train_mask, data.val_mask, data.test_mask, )
                    poison_sum_ensemble.append(poison_acc)
                    res_poison.append(poison_logits)

            if args.ptb_rate == 0:
                type_res.append(res)
                sep_test_acc[feat] = clean_sum_ensemble
                print('clean', sep_test_acc)
            else:
                type_res_evasion.append(res_evasion)
                sep_test_acc_evasion[feat] = evasion_sum_ensemble

                type_res_poison.append(res_poison)
                sep_test_acc_poison[feat] = poison_sum_ensemble

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
            print('============poison==========')
            for key, value in sep_test_acc_poison.items():
                mean = np.mean(value) * 100
                std = np.std(value) * 100
                print(f"{key}: {mean:.2f} ± {std:.2f}")
            ensemble_input = [[type_res_poison[i][j] for i in range(len(feats))] for j in range(len(seeds))]
            ensemble_helper(ensemble_input, data.y, test_masks)


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
    print(accs)
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

