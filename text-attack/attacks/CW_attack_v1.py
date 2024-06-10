import sys
import torch
import numpy as np
from torch import optim
from copy import copy
import ipdb
import time

from utils.util import *


class CarliniL2:

    def __init__(self, search_steps = 1, max_steps = 100, num_classes = 7, device = None):
        logger.info(("const confidence lr:", args.const, args.confidence, args.lr))
        self.device = device

        self.num_classes = num_classes
        self.confidence = args.confidence  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = args.const  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps

        self.mask = None
        self.wv = None
        self.all_dict = None
        self.seq = None
        self.seq_len = None

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            output = np.argmax(output)
        return output != target
    
    def reduce_sum(self, x, keepdim=True):
            # silly PyTorch, when will you get proper reducing sums/means?
            for a in reversed(range(1, x.dim())):
                x = x.sum(a, keepdim=keepdim)
            return x

    def cal_dist(self, x, y, use_l1 = False, keepdim=True):
        if use_l1:
            d = torch.abs(x - y)
        else:
            d = (x - y) ** 2
        return self.reduce_sum(d, keepdim=keepdim)
    
    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]

        loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)
        loss2 = dist.sum()
        
        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var, target_var, scale_const_var):
        # word level attack
        input_adv = modifier_var * self.mask + self.itereated_var

        dist = torch.norm((self.wv - input_adv[:,1:,:].unsqueeze(2)), 2, -1)
        _, new_words_index = torch.min(dist, 2)
        new_index = torch.nn.functional.one_hot(new_words_index)
        new_index = torch.where(new_index==1)
        input_adv[:,1:,:] = self.itereated_var[:,1:,:] = self.wv[new_index].view(input_adv[:,1:,:].shape)
        batch_adv_seq = torch.zeros_like(self.seq).to(self.device)
        batch_adv_seq[:,0] = 101
        batch_adv_seq[:,1:] = self.all_dict[new_index].view(new_words_index.shape)

        output = model(input_ids = self.seq, seq_len = self.seq_len, perturbed=input_adv)['pred']

        # distance to the original input data
        dist = self.cal_dist(input_adv, input_var, use_l1 = args.l1, keepdim=False)

        # get loss and back propagate
        loss = self._loss(output, target_var, dist, scale_const_var)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([modifier_var], args.clip)
        optimizer.step()

        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        output_np = output.data
        input_adv_np = input_adv.data
        return loss_np, dist_np, output_np, input_adv_np, batch_adv_seq

    def run(self, model, input, target, ):
        batch_size = input.size(0) 

        # set the lower and upper bounds accordingly
        scale_const = (torch.ones(batch_size).float() * self.initial_const).to(self.device)
        scale_const.requires_grad = False
        upper_bound = (torch.ones(batch_size) * 1e10).to(self.device)
        upper_bound.requires_grad = False

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        best_embed = input
        self.best_sent = self.seq

        # setup input (image) variable, clamp/scale as necessary
        input_var = torch.tensor(input, requires_grad=False).to(self.device)
        self.itereated_var = input_var

        # setup the target variable, we need it to be in one-hot form for the loss function
        target_var = torch.zeros(target.size() + (self.num_classes,)).to(self.device)
        target_var.scatter_(1, target.unsqueeze(1), 1.)
        target_var.requires_grad = False

        # setup the modifier variable, this is the variable we are optimizing over
        modifier_var = torch.zeros(input_var.size()).float().to(self.device)
        modifier_var.requires_grad = True

        # setup optimizer
        optimizer = optim.Adam([modifier_var], lr=args.lr)

        # search times
        for search_step in range(self.binary_search_steps):
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size
            best_logits = {}
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            # optimize
            for step in range(self.max_steps):
                # perform the attack
                # output 是攻击后的model的test输出  adv_img是输出的词向量矩阵， adv_sents是字的下标组成的list
                loss, dist, output, adv_img, adv_seq = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const,
                    )
                # record
                incorrect = 0
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i].cpu()
                    output_label = torch.argmax(output_logits)
                    di = dist[i]
                    if self._compare(output_logits, target_label) and di < upper_bound[i]:
                        incorrect += 1     
                        if di < best_l2[i]:
                            best_l2[i] = di
                            best_score[i] = output_label
                            best_logits[i] = output_logits
                            best_embed[i] = adv_img[i]
                            self.best_sent[i] = adv_seq[i]

                sys.stdout.flush()
                # end inner step loop
                if incorrect == batch_size:
                    break
        
        return best_embed
