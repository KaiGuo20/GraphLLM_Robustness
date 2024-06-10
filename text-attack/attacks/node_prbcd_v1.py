"""
Robustness of Graph Neural Networks at Scale. NeurIPS 2021.

Modified from https://github.com/sigeisler/robustness_of_gnns_at_scale/blob/main/rgnn_at_scale/attacks/prbcd.py
"""
import numpy as np
from deeprobust.graph.defense_pyg import GCN
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from tqdm import tqdm
import torch_sparse
from torch_sparse import coalesce
import random
import time
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from copy import deepcopy
from torch_geometric.utils import degree
import ipdb



def get_time(f):

    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner


class NodePRBCD:
    def __init__(self, data, model=None, text_encoder = None, sample_choice = "random", mod_nodes = None,
            make_undirected=True,
            eps=1e-7, search_space_size=10_000_000,
            embed_eps = 0, node_search_space_size = 2000,
            max_len = 512, word_embed_size = 384, # 1024; 384
            max_final_samples=20,
            fine_tune_epochs=100,
            epochs=400, 
            lr_adj=0.1, lr_node = .1,
            clip = 0.5, scale_const = 1e8,
            confidence = 0, max_candidates = 10,
            with_early_stopping=True,
            do_synchronize=True,
            device='cuda',
            **kwargs
            ):
        """
        Parameters
        ----------
        data : pyg format data
        model : the model to be attacked, should be models in deeprobust.graph.defense_pyg
        """
        self.device = device
        self.data = data
        self.data.x = self.data.x.to(device)
        self.data.y = self.data.y.to(device)
        self.sample_choice = sample_choice

        if model is None:
            model = self.pretrain_model()
        # try:
        #     self.text_encoder = text_encoder.to(device)
        # except:
        #     self.text_encoder = text_encoder
        self.text_encoder = text_encoder
        self.model = model
        nnodes = data.x.shape[0]
        d = data.x.shape[1]
        self.n, self.d = nnodes, d
        # mod
        print("data.x :", data.x)
        self.feat = self.data.x.detach().cpu()
        self.edge_index = self.data.edge_index.to(device)
        self.edge_weight = None

        # for edges
        self.make_undirected = make_undirected
        self.max_final_samples = max_final_samples
        self.search_space_size = search_space_size
        self.eps = eps
        self.lr_adj = lr_adj
        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2  # We filter self-loops later
        # for nodes
        self.node_search_space_size = node_search_space_size # number of nodes perturbed 
        self.modified_nodes: torch.Tensor = mod_nodes # 1-dim; keep records of perturbed nodes index
        if node_search_space_size:
            self.seqs = data['seqs']
            self.candidates = data['candidates']
            self.word_embeds = self.text_encoder.embedding.weight.to(device)
            # self.word_embeds.requires_grad = False
            self.x = data.x
            self.y = data.y
            self.batch_size = 1
            self.embed_eps = embed_eps # filter nodes with small perturbation
            self.scale_const = scale_const # restrict word embedding
            self.test_nodes = torch.LongTensor(list(self.data.test_mask))
            self.node_tokens = None
            self.iterated_node_tokens = None
            self.current_candidates = None
            self.max_candidates = max_candidates
            self.node_perturbation = None
            self.dist = None
            self.num_class = max(self.y) + 1

            self.mod_nodes = mod_nodes
        

        self.max_len, self.word_embed_size = max_len, word_embed_size
        self.lr_node = lr_node
        self.clip = clip
        self.confidence = confidence

        self.epochs = epochs
        self.epochs_resampling = epochs - fine_tune_epochs

        self.with_early_stopping = with_early_stopping
        self.do_synchronize = do_synchronize

    def pretrain_model(self, model=None):
        data = self.data
        device = self.device
        feat, labels = data.x, data.y
        nclass = max(labels).item()+1

        if model is None:
            model = GCN(nfeat=feat.shape[1], nhid=256, dropout=0,
                    nlayers=3, with_bn=True, weight_decay=5e-4, nclass=nclass,
                    device=device).to(device)
            print(model)

        model.fit(data, train_iters=1000, patience=200, verbose=True) # iter: 1000
        model.eval()
        # model.data = data.to(self.device)
        # ipdb.set_trace()
        output = model.predict(self.data.x.to(self.device), self.edge_index.to(self.device))
        labels = labels.to(device)
        acc = self.get_perf(output, labels, data.test_mask, verbose=0)[1]
        print(f"{model.name} Test set results:", acc)
        self.clean_node_mask = (output.argmax(1) == labels)
        return model, acc, output.detach().cpu()

    def sample_random_block(self, n_perturbations):
        for _ in range(self.max_final_samples):
            self.current_search_space = torch.randint(
                self.n_possible_edges, (self.search_space_size,), device=self.device)
            self.current_search_space = torch.unique(self.current_search_space, sorted=True)
            if self.make_undirected:
                self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = linear_to_full_idx(self.n, self.current_search_space)
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]

            self.perturbed_edge_weight = torch.full_like(
                self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
            )
            if self.current_search_space.size(0) >= n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')
    
    def sample_mod_nodes(self, choice = "random"):
        degrees = degree(self.data.edge_index[0])
        degrees = degrees[self.test_nodes]

        if choice == "random":
            if len(list(self.test_nodes.numpy())) < self.node_search_space_size:
                ext_nodes = torch.LongTensor(random.sample(list(self.data.train_mask), self.node_search_space_size - len(self.test_nodes)))
                self.modified_nodes = torch.cat([self.test_nodes, ext_nodes])
            else:
                self.modified_nodes = torch.LongTensor(random.sample(list(self.test_nodes.numpy()), self.node_search_space_size))
            
        elif choice == "large":
            index = degrees.argsort()[len(degrees) - self.node_search_space_size:] # large first
            self.modified_nodes = self.test_nodes[index.cpu()].cpu()
        
        elif choice == "small":
            index = degrees.argsort()[:self.node_search_space_size] # small first
            self.modified_nodes = self.test_nodes[index.cpu()].cpu()
            
        elif choice == "cluster": # sample one-hop neighbors
            index = degrees.argsort(descending=True) # large or small degree first
            node_index = list(self.test_nodes[index.cpu()].cpu().numpy())

            i = 0
            mod_nodes = set()
            candidates = set(self.test_nodes.numpy())
            candidates = set(node_index)
            while len(mod_nodes) < self.node_search_space_size:
                # new_nodes = random.sample(candidates, 1)
                new_nodes = [node_index[i]]
                i += 1

                neighbors = list(self.edge_index[1][torch.where(self.edge_index[0] == new_nodes[0])[0]].cpu().numpy())

                for nei in neighbors:
                    if len(list(self.edge_index[1][torch.where(self.edge_index[0] == nei)[0]].cpu().numpy())) > 1:
                        # neighbors.remove(nei)
                        pass

                total_new_nodes = new_nodes + neighbors
                mod_nodes.update(total_new_nodes)
                for node in mod_nodes:
                    candidates.discard(node)
            mod_nodes = torch.LongTensor(list(mod_nodes))
            self.modified_nodes = mod_nodes
            self.node_search_space_size = len(mod_nodes)
        
        return None


    def sample_nodes(self):
        # sample modified nodes: random; large; small; cluster
        if self.mod_nodes is None:
            self.sample_mod_nodes(choice = self.sample_choice)
        print("perturbed ratio: ", len(self.modified_nodes) / self.x.shape[0])
        # print(self.modified_nodes)
        
        degrees = degree(self.data.edge_index[0].cpu())
        # print(degrees)
        degrees = degrees[self.modified_nodes]
        
        print("average degree: ", degrees[degrees.argsort()].mean())

        self.node_tokens = torch.LongTensor([])
        self.current_candidates = torch.LongTensor([])
        for idx in self.modified_nodes: 
            seq = torch.tensor(self.seqs[idx], requires_grad=False)
            seq = torch.nn.functional.pad(seq, (0, self.max_len - len(seq)), 'constant', 0)
            self.node_tokens = torch.cat([self.node_tokens, seq.unsqueeze(0)])

            all_candidates = torch.LongTensor([])
            for i in self.seqs[idx]:
                candidates = torch.tensor(self.candidates[idx][i])
                if len(candidates) >= self.max_candidates:
                    candidates = candidates[:self.max_candidates]
                else:
                    candidates = torch.cat([candidates, candidates[0].repeat(self.max_candidates-len(candidates))])
                all_candidates = torch.cat([all_candidates, candidates.unsqueeze(0)])
            self.current_candidates = torch.cat([self.current_candidates, torch.nn.functional.pad(all_candidates, (0, 0, 0, self.max_len - len(all_candidates)), 'constant', 0).unsqueeze(0)])
        # self.node_tokens = self.node_tokens.to(self.device)
        # self.current_candidates = self.current_candidates.to(self.device) 
        self.iterated_node_tokens = self.node_tokens.clone()
        # self.iterated_node_tokens = [item for item in self.node_tokens.clone()]
        self.node_perturbation = torch.full(
                (self.node_search_space_size, self.max_len, self.word_embed_size), self.embed_eps, dtype=torch.float32, requires_grad=True, device = self.device
            )

        self.dist = torch.zeros(self.node_search_space_size).to(self.device)
        

    @torch.no_grad()
    def final_sampling(self, n_perturbations):
        best_loss = -float('Inf')
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0

        _, feat, labels = self.edge_index, self.data.x, self.data.y
        for i in range(self.max_final_samples):
            if best_loss == float('Inf') or best_loss == -float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                print(f'{i}-th sampling: too many samples {n_samples}')
                continue
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            with torch.no_grad():
                output = self.model.forward(feat, edge_index, edge_weight)
                loss = F.nll_loss(output[self.data.val_mask], labels[self.data.val_mask]).item()

            if best_loss < loss:
                best_loss = loss
                print('best_loss:', best_loss)
                best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.device))

        edge_index, edge_weight = self.get_modified_adj()
        edge_mask = edge_weight == 1

        allowed_perturbations = 2 * n_perturbations if self.make_undirected else n_perturbations
        edges_after_attack = edge_mask.sum()
        clean_edges = self.edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    def resample_random_block(self, n_perturbations: int):
        self.keep_heuristic = 'WeightOnly'
        if self.keep_heuristic == 'WeightOnly':
            sorted_idx = torch.argsort(self.perturbed_edge_weight)
            idx_keep = (self.perturbed_edge_weight <= self.eps).sum().long()
            # Keep at most half of the block (i.e. resample low weights)
            if idx_keep < sorted_idx.size(0) // 2:
                idx_keep = sorted_idx.size(0) // 2
        else:
            raise NotImplementedError('Only keep_heuristic=`WeightOnly` supported')

        sorted_idx = sorted_idx[idx_keep:]
        self.current_search_space = self.current_search_space[sorted_idx]
        self.modified_edge_index = self.modified_edge_index[:, sorted_idx]
        self.perturbed_edge_weight = self.perturbed_edge_weight[sorted_idx]

        # Sample until enough edges were drawn
        for i in range(self.max_final_samples):
            n_edges_resample = self.search_space_size - self.current_search_space.size(0)
            lin_index = torch.randint(self.n_possible_edges, (n_edges_resample,), device=self.device)

            self.current_search_space, unique_idx = torch.unique(
                torch.cat((self.current_search_space, lin_index)),
                sorted=True,
                return_inverse=True
            )

            if self.make_undirected:
                self.modified_edge_index = linear_to_triu_idx(self.n, self.current_search_space)
            else:
                self.modified_edge_index = linear_to_full_idx(self.n, self.current_search_space)

            # Merge existing weights with new edge weights
            perturbed_edge_weight_old = self.perturbed_edge_weight.clone()
            self.perturbed_edge_weight = torch.full_like(self.current_search_space, self.eps, dtype=torch.float32)
            self.perturbed_edge_weight[
                unique_idx[:perturbed_edge_weight_old.size(0)]
                ] = perturbed_edge_weight_old # unique_idx: the indices for the old edges

            if not self.make_undirected:
                is_not_self_loop = self.modified_edge_index[0] != self.modified_edge_index[1]
                self.current_search_space = self.current_search_space[is_not_self_loop]
                self.modified_edge_index = self.modified_edge_index[:, is_not_self_loop]
                self.perturbed_edge_weight = self.perturbed_edge_weight[is_not_self_loop]

            if self.current_search_space.size(0) > n_perturbations:
                return
        raise RuntimeError('Sampling random block was not successfull. Please decrease `n_perturbations`.')
    
    def resample_nodes(self,):
        # resample nodes
        # changed: self.node_tokens, self.iterated_node_tokens and self.modified_nodes
        #    1. filter nodes with less perturbation based on self.embed_eps. embed_eps is defined by the smallest perturbation possible
        # or 2. 

        # based on embedding perturbation
        index = torch.where(self.dist < self.embed_eps)[0]
        # based on label perturbation

        if len(index) == 0:
            return
        if self.n / 2 < len(index):
            index = index[int(self.n / 2):] # Keep at most half of the block
        res_n = list(set(self.test_nodes) - set(self.modified_nodes.numpy()))
        new_nodes = torch.LongTensor(random.sample(res_n, len(index)))
        self.modified_nodes[index] = new_nodes
        self.node_tokens.data[index] = self.iterated_node_tokens.data[index] = torch.tensor(self.seqs[self.modified_nodes[index.cpu()]], requires_grad = False).to(self.device)
        self.node_perturbation[index] = 0
        
        new_candidates = torch.LongTensor([])
        for idx in new_nodes: 
            seq = torch.tensor(self.seqs[idx], requires_grad=False)
            seq = torch.nn.functional.pad(seq, (0, self.max_len - len(seq)), 'constant', 0)
            self.node_tokens = torch.cat([self.node_tokens, seq.unsqueeze(0)])

            all_candidates = torch.LongTensor([])
            for i in self.seqs[idx]:
                candidates = torch.tensor(self.candidates[idx][i])
                if len(candidates) >= self.max_candidates:
                    candidates = candidates[:self.max_candidates]
                else:
                    candidates = torch.cat([candidates, candidates[0].repeat(self.max_candidates-len(candidates))])
                all_candidates = torch.cat([all_candidates, candidates.unsqueeze(0)])
            new_candidates = torch.cat([new_candidates, torch.nn.functional.pad(all_candidates, (0, 0, 0, self.max_len - len(all_candidates)), 'constant', 0).unsqueeze(0)])
        self.current_candidates[index] = new_candidates

        self.dist = torch.zeros_like(self.node_tokens)

    def project(self, n_perturbations, values, eps, inplace=False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values
    
    def get_modified_adj(self):
        if self.make_undirected:
            modified_edge_index, modified_edge_weight = to_symmetric(
                self.modified_edge_index, self.perturbed_edge_weight, self.n
            )
        else:
            modified_edge_index, modified_edge_weight = self.modified_edge_index, self.perturbed_edge_weight

        edge_index = torch.cat((self.edge_index.to(self.device), modified_edge_index), dim=-1) #  save perturbed edge
        edge_weight = torch.cat((self.edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=self.n, n=self.n, op='sum')

        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        return edge_index, edge_weight

    @get_time
    def optimize_feats(self, labels, edge_index, edge_weight):
        self.node_perturbation.requires_grad = False
        
        data = Data(x = self.feat, y = labels, edge_index= edge_index, edge_weight = edge_weight)

        loader = NeighborLoader(data, 
                                input_nodes = self.modified_nodes,
                                num_neighbors = [-1,-1],
                                batch_size = self.batch_size,
                                replace=False,
                                shuffle=False,)
        n = 0
        for i, batch in enumerate(loader):
            
            # print("-1", torch.cuda.memory_allocated()/1024**2)
            batch_size = len(batch.input_id)
            node_idx = batch.n_id[:batch_size] # batch
            index = torch.nonzero(torch.isin(self.modified_nodes, node_idx)).view(-1)

            batch_seqs = self.iterated_node_tokens[index] # batch * 512
            # batch_seqs = batch_seqs.to(self.device)
            batch_candidates = self.current_candidates[index] # batch * 512 * 10
            batch_candidates = batch_candidates.to(self.device)
            batch_labels = labels[node_idx]
            # 
            # get perturbation
            perturbation = self.node_perturbation.data[index]
        
            input_embeds = self.word_embeds[batch_seqs.view(-1).to(self.device)].view(batch_size, self.max_len, self.word_embed_size) # batch * 512 * 368
            # input_embeds.requires_grad = False
            batch_embeds = torch.tensor(input_embeds)
            batch_candidates_embeds = self.word_embeds[batch_candidates.view(-1).to(self.device)].view(batch_size, self.max_len, self.max_candidates, self.word_embed_size) # batch * 512 * 10 * 368
            mask = torch.zeros_like(batch_seqs)
            mask[torch.where(batch_seqs != 0)] = 1
            mask = mask.unsqueeze(2).repeat(1,1,self.word_embed_size) # batch * 512 * 384

            seq_lens = []
            for seq in self.iterated_node_tokens[index]:
                seq_lens.append(len(torch.where(seq!=0)[0]))
            seq_lens = torch.LongTensor(seq_lens)
            
            perturbation.requires_grad = True
            sem_optimizer = torch.optim.Adam([{'params': perturbation, 'lr': self.lr_node}])
            # 
            # print("0", torch.cuda.memory_allocated()/1024**2) 
            for step in range(4):
                
                input_adv = batch_embeds + mask.to(self.device) * perturbation # batch * 512 * 384
                # find and project
                dist = torch.norm((batch_candidates_embeds - input_adv.unsqueeze(2)), 2, -1)
                _, new_words_index = torch.min(dist, 2) 
                del dist
                new_index = torch.nn.functional.one_hot(new_words_index)
                new_index = torch.where(new_index==1)
                

                # ori_input_adv = batch_embeds + mask * perturbation
                input_adv.data = batch_embeds.data = batch_candidates_embeds.data[new_index].view(input_adv.shape)
                # ipdb.set_trace()
                self.iterated_node_tokens[index] = batch_candidates.data[new_index].view(new_words_index.shape).cpu()
                del new_index
                del new_words_index
                
                # print("diff tokens: ", (self.node_tokens != self.iterated_node_tokens.data).sum())
                # print("diff embeds: ", (input_adv != ori_input_adv).sum())
                # print(input_adv)
                # ipdb.set_trace()
                feats = self.text_encoder.generate(input_ids = self.iterated_node_tokens[index], seq_len = seq_lens, perturbed = input_adv)
                feats = feats.float()
                # feats.retain_grad()
                # print(self.feat)
                # ipdb.set_trace()
                try:
                    all_feat = self.feat.clone().to(self.device)
                except:
                    all_feat = self.feat.clone().cuda()
                # ipdb.set_trace()
                all_feat[node_idx] = feats
                
                original_embeds = self.word_embeds[self.node_tokens[index]] # [0][:seq_lens.item()].unsqueeze(0)
                dist = ((original_embeds - input_adv)**2).sum(2).sum(1)
                # self.dist.data[index] = dist.data
                # fake_weight = torch.zeros_like(batch.edge_index[0]).to(torch.float32)

                output = self.model.predict(all_feat, batch.edge_index.detach(), batch.edge_weight.detach()) # batch.edge_index.detach(), batch.edge_weight.detach()
    
                output = output[node_idx]
                batch_labels = labels[node_idx]
                # dist=None

                if torch.sum(output.argmax(1) == batch_labels) == 0:
                    break

                sem_loss = self._loss(output, batch_labels, dist, self.scale_const) # loss from semattack  

                sem_optimizer.zero_grad()
                # ipdb.set_trace()
                sem_loss.backward()
                

                torch.nn.utils.clip_grad_norm_([perturbation], self.clip)
                sem_optimizer.step()
                

            self.node_perturbation.data[index] = perturbation.data
            self.feat = all_feat.detach().cpu()
            del all_feat

    def attack(self, edge_index=None, edge_weight=None, edge_ptb_rate=0.1):
        data = self.data
        epochs, lr_adj, lr_node = self.epochs, self.lr_adj, self.lr_node
        model = self.model
        model.eval() # should set to eval

        self.edge_index, feat, labels = data.edge_index, data.x, data.y
        with torch.no_grad():
            output = model.predict(feat, self.edge_index, self.edge_weight)
            pred = output.argmax(1)
        gt_labels = labels
        labels = labels.clone() # to avoid shallow copy
        
        labels[data.test_mask] =  pred[data.test_mask]
        labels[data.val_mask] =  pred[data.val_mask]

        if edge_index is not None:
            self.edge_index = edge_index

        self.edge_weight = torch.ones(self.edge_index.shape[1]).to(self.device)

        n_perturbations = int(edge_ptb_rate * self.edge_index.shape[1] //2)
        print('n_perturbations:', n_perturbations)
        self.sample_random_block(n_perturbations)
        if self.node_search_space_size:
            self.sample_nodes()

        self.perturbed_edge_weight.requires_grad = True
        self.prbcd_optimizer = torch.optim.Adam([
                                            {'params': self.perturbed_edge_weight, 'lr': lr_adj},
                                            ])
        if edge_ptb_rate:
            edge_index, edge_weight  = self.get_modified_adj()
        else:
            edge_index, edge_weight  = self.edge_index, self.edge_weight
        # 攻击多轮
        for it in tqdm(range(epochs)):
            # t1 = time.time()
            # perturbing features 文本攻击
            if it % 4 == 0 and self.node_search_space_size:
                print("perturbing features")
                # ipdb.set_trace()
                self.optimize_feats(labels, edge_index.detach(), edge_weight)
            # perturbing graph structure 图结构攻击
            if edge_ptb_rate:
                print("perturbing graph structure")
                self.perturbed_edge_weight.requires_grad = True

                if torch.cuda.is_available() and self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                output = model.forward(self.feat.detach().to(self.device), edge_index, edge_weight)
                prbcd_loss = self.loss_attack(output, labels, type='tanhMargin') # loss from prbcd
                gradients = self.grad_with_checkpoint(prbcd_loss) # only needed for prbcd loss

                if torch.cuda.is_available() and self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                if it % 10 == 0:
                    print(f'Epoch {it}: prbcd {prbcd_loss}')

                with torch.no_grad():
                    self.update_params(gradients)
                    self.perturbed_edge_weight = self.project(n_perturbations, self.perturbed_edge_weight, self.eps) 
                    del edge_index, edge_weight

                    if it < self.epochs_resampling - 1:
                        self.resample_random_block(n_perturbations)
                        if self.node_search_space_size:
                            self.resample_nodes()
                    # if it % 100 == 0:
                    #     edge_index, edge_weight = self.get_modified_adj()
                    #     feat = self.get_modified_feat()
                    #     output = model.predict(feat, edge_index, edge_weight)
                    #     loss_val = F.nll_loss(output[data.val_mask], labels[data.val_mask])
                    #     print("val loss:", loss_val)

                self.perturbed_edge_weight.requires_grad = True
                self.prbcd_optimizer = torch.optim.Adam([
                                                {'params': self.perturbed_edge_weight, 'lr': lr_adj},
                                                ])
                edge_index, edge_weight  = self.get_modified_adj()
            #
            # t2 = time.time()
            # print(f"iter: {t2 - t1}")
        if edge_ptb_rate == 0:
            edge_index, edge_weight = self.edge_index, self.edge_weight
        else:
            # Sample final discrete graph
            edge_index, edge_weight = self.final_sampling(n_perturbations)
        # print("check difference:", torch.sum(self.feat != self.data.x))
        output = model.predict(self.feat.to(self.device), edge_index.to(self.device), edge_weight)
        # print("final: ", output.argmax(1)[self.modified_nodes])
        # output = model.forward(self.feat)
        print('Test:')
        self.get_perf(output, gt_labels, data.test_mask)
        print('Validatoin:')
        self.get_perf(output, gt_labels, data.val_mask)
        if self.node_search_space_size:
            # perturbed_seqs = [seq.cpu() for seq in self.iterated_node_tokens]
            perturbed_seqs = self.iterated_node_tokens
        else:
            perturbed_seqs = None

        return self.feat, self.modified_nodes, perturbed_seqs, edge_index, edge_weight
    

            
    def update_params(self, gradients):
        self.prbcd_optimizer.zero_grad()
        self.perturbed_edge_weight.grad = -gradients[0]
        self.prbcd_optimizer.step()
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = self.eps

    def loss_attack(self, logits, labels, type='CE'):
        self.loss_type = type
        if self.loss_type == 'tanhMargin':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            margin = (
                logits[np.arange(logits.size(0)), labels]
                - logits[np.arange(logits.size(0)), best_non_target_class]
            )
            loss = torch.tanh(-margin).mean()
        elif self.loss_type == 'MCE':
            not_flipped = logits.argmax(-1) == labels
            loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
        elif self.loss_type == 'NCE':
            sorted = logits.argsort(-1)
            best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
            loss = -F.cross_entropy(logits, best_non_target_class)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss
    
    # This loss is adopted from SemAttack: https://github.com/AI-secure/SemAttack/blob/main/English/bert/CW_attack.py
    def _loss(self, output, target, dist = None, scale_const = None):
        # compute the probability of the label class versus the maximum other
        
        target = torch.nn.functional.one_hot(target, self.num_class)

        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]

        loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)
        if dist is not None:
            loss2 = dist.sum()
        else:
            loss2 = 0
        
        loss = loss1 + loss2
        return loss

    def get_perf(self, output, labels, mask, verbose=True):
        loss = F.nll_loss(output[mask], labels[mask])
        acc = utils.accuracy(output[mask], labels[mask])
        if verbose:
            print("loss= {:.4f}".format(loss.item()),
                  "accuracy= {:.4f}".format(acc.item()))
        return loss.item(), acc.item()
    
    def grad_with_checkpoint(self, prbcd_loss): 
        inputs = tuple([self.perturbed_edge_weight])
        for input in inputs:
            if not input.is_leaf:
                input.retain_grad()
        torch.autograd.backward(prbcd_loss)

        grad_outputs = []
        for input in inputs:
            grad_outputs.append(input.grad.clone())
            input.grad.zero_()
        return grad_outputs

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(torch.exp(x) * x).sum(1)

def to_symmetric(edge_index, edge_weight, n, op='mean'):
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight

def linear_to_full_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = lin_idx // n
    col_idx = lin_idx % n
    return torch.stack((row_idx, col_idx))

def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (
        n
        - 2
        - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + 1 - n * (n - 1) // 2
        + (n - row_idx) * ((n - row_idx) - 1) // 2
    )
    return torch.stack((row_idx, col_idx))



def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= epsilon):
            break
    return miu


# if __name__ == "__main__":
#     from ogb.nodeproppred import PygNodePropPredDataset
#     from torch_geometric.utils import to_undirected
#     import torch_geometric.transforms as T
#     dataset = PygNodePropPredDataset(name='ogbn-arxiv')
#     dataset.transform = T.NormalizeFeatures()
#     data = dataset[0]
#     if not hasattr(data, 'train_mask'):
#         utils.add_mask(data, dataset)
#     data.edge_index = to_undirected(data.edge_index, data.num_nodes)
#     agent = PRBCD(data)
#     edge_index, edge_weight = agent.attack()