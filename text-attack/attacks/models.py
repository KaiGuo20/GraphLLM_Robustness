import numpy as np
from torch import nn
import torch
from pytorch_transformers.modeling_bert import *
from sentence_transformers import SentenceTransformer
from pytorch_transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn.models import MLP
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_sparse import coalesce, SparseTensor, matmul
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from deeprobust.graph import utils
import torch
from sklearn.feature_extraction.text import CountVectorizer
import pdb
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def seq_len_to_mask(seq_len, max_len=None):
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, "seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        if max_len is None:
            max_len = int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.LongTensor) or isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, "seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        if max_len is None:
            max_len = seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.LongTensor.")

    return mask

class BertEmbeddings_attack(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, word_embedding, token_type_embedding, position_embedding, layer_norm, dropout):
        super(BertEmbeddings_attack, self).__init__()
        self.word_embeddings = word_embedding
        self.position_embeddings = position_embedding
        self.token_type_embeddings = token_type_embedding
        self.LayerNorm = layer_norm
        self.dropout = dropout

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(self, input_ids = None, token_type_ids=None, position_ids=None, perturbed=None, seq_len = None):
        if position_ids is None:
            if seq_len is not None:
                position_ids = self.position_ids[:, 0 : seq_len]
            else:
                position_ids = self.position_ids[:, 0 : input_ids.size()[1]]

        
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if perturbed is not None:
            embeddings = perturbed + position_embeddings + token_type_embeddings
        else:
            words_embeddings = self.word_embeddings(input_ids)
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel_attack(nn.Module):
    def __init__(self, model_name, sbert, device):
        super(BertModel_attack, self).__init__()
        self.device = device
        # self.config = list(sbert._modules['0'].modules())[1].encoder.config
        self.config = sbert.config

        self.model_name = model_name

        word_embedding_module = sbert.embeddings.word_embeddings
        position_embedding_module = sbert.embeddings.position_embeddings
        token_type_embedding_module = sbert.embeddings.token_type_embeddings
        layer_norm = sbert.embeddings.LayerNorm
        dropout = sbert.embeddings.dropout

        # word_embedding_module = list(sbert._modules['0'].modules())[1].embeddings.word_embeddings
        # position_embedding_module = list(sbert._modules['0'].modules())[1].embeddings.position_embeddings
        # token_type_embedding_module = list(sbert._modules['0'].modules())[1].embeddings.token_type_embeddings
        # layer_norm = list(sbert._modules['0'].modules())[1].embeddings.LayerNorm
        # dropout = list(sbert._modules['0'].modules())[1].embeddings.dropout
        self.embeddings = BertEmbeddings_attack(self.config, word_embedding_module, token_type_embedding_module, position_embedding_module, layer_norm, dropout)
        # self.encoder = list(sbert._modules['0'].modules())[1].encoder
        # self.pooler = list(sbert._modules['0'].modules())[1].pooler

        self.encoder = sbert.encoder
        self.pooler = sbert.pooler
        # self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids = None, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
                perturbed=None, size = None):
        if size is not None:
            batch_size, seq_length = size[0], size[1]
        else:
            batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=self.device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long, device=self.device)
        
        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                        perturbed=perturbed, seq_len = seq_length)
    
        encoder_outputs = self.encoder(embedding_output,
                                    extended_attention_mask,
                                    head_mask=head_mask, return_dict = False)
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)


        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs, attention_mask


class BertC(nn.Module):
    def __init__(self, model_name = "sbert", dropout=0.5, num_class=7, ft_data = None, device = None):
        super(BertC, self).__init__()
        self.device = device
        self.num_class = num_class
        self.max_len = 512
        self.model_name = model_name
        # self.core_mdoel = SentenceTransformer('sbert', device='cuda').to('cuda')
        print(model_name)
        if model_name != "llama":
            self.core_model = AutoModel.from_pretrained(model_name).to(self.device)
            self.base_model = SentenceTransformer('sbert', device='cuda').to(self.device)
            self.model = BertModel_attack(model_name, self.core_model, device).to(self.device)
            # ipdb.set_trace()
            self.embedding = self.core_model.embeddings.word_embeddings
            self.pooling = list(self.base_model._modules['1'].modules())[0]
            self.normalize = list(self.base_model._modules['2'].modules())[0]
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.core_model = AutoModelForCausalLM.from_pretrained(
                                                        model_name,  # Llama 2 7B, same as before
                                                        device_map='cuda:0',
                                                        quantization_config=bnb_config,  # Same quantization config as before
                                                    )
            # ipdb.set_trace()
            self.embedding = self.core_model.base_model.embed_tokens
                
            if ft_data is not None:
                self.core_model = PeftModel.from_pretrained(self.core_model, f"llama-finetune/{ft_data}/checkpoint-500")

        # self.tokenizer = self.core_mdoel._first_module()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 512)
        for param in self.core_model.parameters():
            param.requires_grad = False

        self.loss_f = nn.CrossEntropyLoss()
        
        self.mlp = MLP(channel_list=[384, 256, num_class], dropout=dropout, norm=None)


    def forward(self, txt = None, input_ids = None, seq_len = None, gold=None, perturbed=None, size = None):
        if input_ids is not None:
            attention_mask = seq_len_to_mask(seq_len, input_ids.size(1)).to(self.device)
            token_type_ids = None
            input_ids = input_ids.to(self.device)
        elif txt is not None:
            input_ids, token_type_ids, attention_mask = self.tokenizer.tokenize(txt).values()
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
        else:
            attention_mask = None
            token_type_ids = None
        
        out, attention_mask = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, perturbed=perturbed, size = size)
    
        dit = {'input_ids': input_ids, "token_embeddings":out[0], "attention_mask":attention_mask}
        embed = self.normalize(self.pooling(dit))['sentence_embedding']
        
        logits = self.mlp(embed)
        ret = {"pred": logits}
        if gold is not None:
            ret["loss"] = self.loss_f(logits, gold)
        ret['embedding'] = out[0]
        return ret
    
    def generate(self, txt = None, input_ids = None, seq_len = None, perturbed = None, size = None):
        if input_ids is not None:
            
            attention_mask = seq_len_to_mask(seq_len, len(input_ids[0])).to(self.device)
            token_type_ids = None
            # ipdb.set_trace()
            input_ids = input_ids.to(self.device)
        elif txt is not None:
            input_ids, token_type_ids, attention_mask = self.tokenizer(txt, truncation = True).values()
            token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
            input_ids = torch.LongTensor(input_ids).to(self.device)
            attention_mask = torch.LongTensor(attention_mask).to(self.device)
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
                token_type_ids = token_type_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)     
        else:
            attention_mask = None
            token_type_ids = None
            
        if self.model_name == "llama":
            # ipdb.set_trace()
            if perturbed is not None:
                result = self.core_model(inputs_embeds = perturbed, output_hidden_states=True)
            else:
                result = self.core_model(input_ids = input_ids, output_hidden_states=True)
            # print(self.core_model.device)
            try:
                embed = result.hidden_states[-1].mean(axis=1)
            except:
                embed = result['last_hidden_state'].mean(axis=1)
        else:
            out, attention_mask = self.model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, perturbed=perturbed, size = size)
            dit = {'input_ids': input_ids, "token_embeddings":out[0], "attention_mask":attention_mask}
            embed = self.normalize(self.pooling(dit))['sentence_embedding']
        return embed
    
    def get_word_embeds(self, seq):
        seq = torch.nn.functional.pad(torch.LongTensor(seq), (0, self.max_len - len(seq)), 'constant', 0)
        embed = self.model.embeddings.word_embeddings(seq.to(self.device))
        return embed
    
import ipdb

class BOW(nn.Module):
    def __init__(self, dataset, vocab_size, split, device):
        super(BOW, self).__init__()
        training_txt = [txt for i, txt in enumerate(dataset.raw_texts) if i in torch.where(dataset.train_masks[split]==True)[0]]
        self.device = device
        self.counter = CountVectorizer()
        self.counter.fit(dataset.raw_texts)
        self.tokenizer = self.counter.build_tokenizer()
        
        self.vocab = self.counter.vocabulary_
        feature = self.counter.fit_transform(training_txt)
        count = torch.LongTensor(feature.toarray().sum(0))
        count_index = count.argsort(descending = True)
        tmp_vocab = dict([(value, key) for key, value in self.counter.vocabulary_.items()])
        self.words = [tmp_vocab[idx.item()].lower() for idx in count_index]
        self.words = self.words[:vocab_size]
        self.embedding = nn.Embedding(len(self.vocab)+1, vocab_size)
        self.embedding.weight.data = torch.zeros_like(self.embedding.weight)
        for i in range(vocab_size):
            # ipdb.set_trace()
            idx = self.vocab[self.words[i].lower()]
            self.embedding.weight.data[idx+1] = 0
            self.embedding.weight.data[idx+1][i] = 1
        self.vocab = dict([(key, value+1) for key, value in self.vocab.items()])

        self.embedding = self.embedding.to(device)



    def generate(self, input_ids = None, seq_len = None, perturbed = None):
        # inputs should be a batch of sentences with word indices
        if perturbed is None:
            embeddings = self.embedding(input_ids.to(self.device))  # Convert word indices to embeddings
        else:
            embeddings = perturbed
        bow_features = embeddings.sum(dim=1)  # Sum pooling to aggregate word embeddings
        # Further processing can be done here
        return bow_features




def get_sbert_embedding(texts):
    sbert_model = SentenceTransformer('sbert', device='cuda').to('cuda')
    sbert_embeds = sbert_model.encode(texts, batch_size=5, show_progress_bar=True)
    return sbert_model, sbert_embeds


#-------------------mlp--------------------------------------

from torch_geometric.nn.models import MLP
import torch
import torch.nn.functional as F
from copy import deepcopy
import torch
from ogb.nodeproppred import Evaluator


class UniversalMLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None, return_embeds = False, device='cpu') -> None:
        super().__init__()
        self.device = device
        hidden_dimensions = [hidden_dimension] * (num_layers - 1)
        self.hidden_dimensions = [input_dim] + hidden_dimensions + [num_classes]
        self.dropout = dropout
        self.norm = norm
        self.mlp = MLP(channel_list=self.hidden_dimensions, dropout=self.dropout, norm=self.norm).to(self.device)
        self.return_embeds = False
    
    def reset_params(self):
        self.mlp = MLP(channel_list=self.hidden_dimensions, dropout=self.dropout, norm=self.norm).to(self.device)

    def forward(self, x, **kwargs):
        return self.mlp(x)


class mlp_model():
    def __init__(self, input_dim, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UniversalMLP(num_layers = 2, input_dim = input_dim, hidden_dimension = 256, num_classes = num_classes, dropout = 0.5, norm = None, return_embeds = True).to(self.device)
        self.normalize = False
        self.early_stop_start = 400
        self.early_stopping = 10

        self.output = None
        self.hidden_sizes = [256]
        self.nfeat = input_dim
        self.nclass = num_classes
        self.with_relu = True

    def train(self, model, data, optimizer, loss_fn, train_mask, val_mask):
        model.train()
        optimizer.zero_grad()
        preds = model(data.x.to(self.device))
        if len(data.y.shape) != 1:
            y = data.y.squeeze(1).to(self.device)
        else:
            y = data.y.to(self.device)
        train_loss = loss_fn(preds[train_mask], y[train_mask])
        train_loss.backward()
        optimizer.step()
        val_loss = loss_fn(preds[val_mask], y[val_mask])
        val_acc, output = self.test(model, data, val_mask)
        return train_loss, val_loss, val_acc
    
    @torch.no_grad()
    def test(self, model, data, mask):
        model.eval()
        # model.model.initialized = False
        out = model(data.x.to(self.device))
        y_pred = out.argmax(dim=-1, keepdim=True)

        if len(data.y.shape) == 1:
            y = data.y.unsqueeze(dim=1).to(self.device)  # for non ogb datas
        else:
            y = data.y.to(self.device)

        evaluator = Evaluator(name='ogbn-arxiv')
        acc = evaluator.eval({
            'y_true': y[mask],
            'y_pred': y_pred[mask],
        })['acc']

        return acc, out
    
    
    def fit(self, data):
        self.model.reset_params()
        early_stop_accum = 0
        best_val = 0
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.01, weight_decay = 5e-4)

        loss_fn = torch.nn.CrossEntropyLoss() 
        
        if self.normalize:
            data.x = F.normalize(data.x, dim = -1)
        data = data.to(self.device)

        for i in range(300):
            train_mask = data.train_mask
            val_mask = data.val_mask
            train_loss, val_loss, val_acc = self.train(self.model, data, optimizer, loss_fn, train_mask, val_mask)
            # print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc}")
            if val_acc > best_val:
                best_val = val_acc
                self.best_model = deepcopy(self.model)
                early_stop_accum = 0
            else:
                if i >= self.early_stop_start:
                    early_stop_accum += 1
                if early_stop_accum > self.early_stopping and i >= self.early_stop_start:
                    break

        test_acc, res = self.test(self.best_model, data, data.test_mask)

        self.output = res.to(self.device)
        print(f"Test ACC: {test_acc}")

        return test_acc


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    # def fit(self, pyg_data, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
    #     if initialize:
    #         self.initialize()

    #     # self.data = pyg_data[0].to(self.device)
    #     self.data = pyg_data.to(self.device)
    #     # By default, it is trained with early stopping on validation
    #     self.train_with_early_stopping(train_iters, patience, verbose)

    # def finetune(self, edge_index, edge_weight, feat=None, train_iters=10, verbose=True):
    #     if verbose:
    #         print(f'=== finetuning {self.name} model ===')
    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     labels = self.data.y
    #     if feat is None:
    #         x = self.data.x
    #     else:
    #         x = feat
    #     train_mask, val_mask = self.data.train_mask, self.data.val_mask
    #     best_loss_val = 100
    #     best_acc_val = 0
    #     for i in range(train_iters):
    #         self.train()
    #         optimizer.zero_grad()
    #         output = self.forward(x, edge_index, edge_weight)
    #         loss_train = F.nll_loss(output[train_mask], labels[train_mask])
    #         loss_train.backward()
    #         optimizer.step()

    #         if verbose and i % 50 == 0:
    #             print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

    #         self.eval()
    #         with torch.no_grad():
    #             output = self.forward(x, edge_index)
    #         loss_val = F.nll_loss(output[val_mask], labels[val_mask])
    #         acc_val = utils.accuracy(output[val_mask], labels[val_mask])

    #         # if best_loss_val > loss_val:
    #         #     best_loss_val = loss_val
    #         #     best_output = output
    #         #     weights = deepcopy(self.state_dict())

    #         if best_acc_val < acc_val:
    #             best_acc_val = acc_val
    #             best_output = output
    #             weights = deepcopy(self.state_dict())

    #     print('best_acc_val:', best_acc_val.item())
    #     self.load_state_dict(weights)
    #     return best_output


    # def _fit_with_val(self, pyg_data, train_iters=1000, initialize=True, verbose=False, **kwargs):
    #     if initialize:
    #         self.initialize()

    #     # self.data = pyg_data[0].to(self.device)
    #     self.data = pyg_data.to(self.device)
    #     if verbose:
    #         print(f'=== training {self.name} model ===')
    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    #     labels = self.data.y
    #     train_mask, val_mask = self.data.train_mask, self.data.val_mask

    #     x, edge_index = self.data.x, self.data.edge_index
    #     for i in range(train_iters):
    #         self.train()
    #         optimizer.zero_grad()
    #         output = self.forward(x, edge_index)
    #         loss_train = F.nll_loss(output[train_mask+val_mask], labels[train_mask+val_mask])
    #         loss_train.backward()
    #         optimizer.step()

    #         if verbose and i % 50 == 0:
    #             print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

    # def fit_with_val(self, pyg_data, train_iters=1000, initialize=True, patience=100, verbose=False, **kwargs):
    #     if initialize:
    #         self.initialize()

    #     self.data = pyg_data.to(self.device)
    #     self.data.train_mask = self.data.train_mask + self.data.val1_mask
    #     self.data.val_mask = self.data.val2_mask
    #     self.train_with_early_stopping(train_iters, patience, verbose)

    # def train_with_early_stopping(self, train_iters, patience, verbose):
    #     """early stopping based on the validation loss
    #     """
    #     if verbose:
    #         print(f'=== training {self.name} model ===')
    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    #     labels = self.data.y
    #     train_mask, val_mask = self.data.train_mask, self.data.val_mask

    #     early_stopping = patience
    #     best_loss_val = 100
    #     best_acc_val = 0
    #     best_epoch = 0

    #     x, edge_index = self.data.x, self.data.edge_index
    #     for i in range(train_iters):
    #         self.train()
    #         optimizer.zero_grad()

    #         output = self.forward(x, edge_index)

    #         loss_train = F.nll_loss(output[train_mask], labels[train_mask])
    #         loss_train.backward()
    #         optimizer.step()

    #         if verbose and i % 50 == 0:
    #             print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

    #         self.eval()
    #         output = self.forward(x, edge_index)
    #         loss_val = F.nll_loss(output[val_mask], labels[val_mask])
    #         acc_val = utils.accuracy(output[val_mask], labels[val_mask])
    #         # print(acc)

    #         # if best_loss_val > loss_val:
    #         #     best_loss_val = loss_val
    #         #     self.output = output
    #         #     weights = deepcopy(self.state_dict())
    #         #     patience = early_stopping
    #         #     best_epoch = i
    #         # else:
    #         #     patience -= 1

    #         if best_acc_val < acc_val:
    #             best_acc_val = acc_val
    #             self.output = output
    #             weights = deepcopy(self.state_dict())
    #             patience = early_stopping
    #             best_epoch = i
    #         else:
    #             patience -= 1

    #         if i > early_stopping and patience <= 0:
    #             break

    #     if verbose:
    #          # print('=== early stopping at {0}, loss_val = {1} ==='.format(best_epoch, best_loss_val) )
    #          print('=== early stopping at {0}, acc_val = {1} ==='.format(best_epoch, best_acc_val) )
    #     self.load_state_dict(weights)

    # def test(self):
    #     """Evaluate model performance on test set.
    #     Parameters
    #     ----------
    #     idx_test :
    #         node testing indices
    #     """
    #     self.eval()
    #     test_mask = self.data.test_mask
    #     labels = self.data.y
    #     output = self.forward(self.data.x, self.data.edge_index)
    #     # output = self.output
    #     loss_test = F.nll_loss(output[test_mask], labels[test_mask])
    #     acc_test = utils.accuracy(output[test_mask], labels[test_mask])
    #     print("Test set results:",
    #           "loss= {:.4f}".format(loss_test.item()),
    #           "accuracy= {:.4f}".format(acc_test.item()))
    #     return acc_test.item()

    def predict(self, x=None, edge_index=None, edge_weight=None):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities)
        """
        self.eval()
        if x is None or edge_index is None:
            x, edge_index = self.data.x, self.data.edge_index
        return self.forward(x, edge_index, edge_weight)

    # def _ensure_contiguousness(self,
    #                            x,
    #                            edge_idx,
    #                            edge_weight):
    #     if not x.is_sparse:
    #         x = x.contiguous()
    #     if hasattr(edge_idx, 'contiguous'):
    #         edge_idx = edge_idx.contiguous()
    #     if edge_weight is not None:
    #         edge_weight = edge_weight.contiguous()
    #     return x, edge_idx, edge_weight


class new_MLP(BaseModel):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None):

        super(new_MLP, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.with_bn = with_bn
        self.name = 'MLP'
  
        self.normalize = False
        self.early_stop_start = 400
        self.early_stopping = 10

        self.output = None
        self.hidden_sizes = [nhid]
        self.nfeat = nfeat
        self.nclass = nclass
        self.with_relu = True

        self.model = UniversalMLP(num_layers = nlayers, input_dim = nfeat, hidden_dimension = nhid, num_classes = nclass, dropout = dropout, norm = None, return_embeds = True, device = self.device).to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        return self.model(x)
    
    def train_model(self, model, data, optimizer, loss_fn, train_mask, val_mask):
        model.train()
        optimizer.zero_grad()
        preds = model(data.x.to(self.device))
        if len(data.y.shape) != 1:
            y = data.y.squeeze(1).to(self.device)
        else:
            y = data.y.to(self.device)
        train_loss = loss_fn(preds[train_mask], y[train_mask])
        train_loss.backward()
        optimizer.step()
        val_loss = loss_fn(preds[val_mask], y[val_mask])
        val_acc, output = self.test(model, data, val_mask)
        return train_loss, val_loss, val_acc
    
    @torch.no_grad()
    def test(self, model, data, mask):
        model.eval()
        # model.model.initialized = False
        out = model(data.x.to(self.device))
        y_pred = out.argmax(dim=-1, keepdim=True)

        if len(data.y.shape) == 1:
            y = data.y.unsqueeze(dim=1).to(self.device)  # for non ogb datas
        else:
            y = data.y.to(self.device)

        evaluator = Evaluator(name='ogbn-arxiv')
        acc = evaluator.eval({
            'y_true': y[mask],
            'y_pred': y_pred[mask],
        })['acc']

        return acc, out
    
    
    def fit(self, data, train_iters=1000, patience=200, verbose=True):
        self.model.reset_params()
        early_stop_accum = 0
        best_val = 0
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.01, weight_decay = 5e-4)

        loss_fn = torch.nn.CrossEntropyLoss() 
        
        if self.normalize:
            data.x = F.normalize(data.x, dim = -1)
        data = data.to(self.device)

        for i in range(300):
            train_mask = data.train_mask
            val_mask = data.val_mask
            train_loss, val_loss, val_acc = self.train_model(self.model, data, optimizer, loss_fn, train_mask, val_mask)
            # print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc}")
            if val_acc > best_val:
                best_val = val_acc
                self.best_model = deepcopy(self.model)
                early_stop_accum = 0
            else:
                if i >= self.early_stop_start:
                    early_stop_accum += 1
                if early_stop_accum > self.early_stopping and i >= self.early_stop_start:
                    break

        test_acc, res = self.test(self.best_model, data, data.test_mask)

        self.output = res.to(self.device)
        print(f"Test ACC: {test_acc}")

        return test_acc

    """
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import SGConv

class SGC(torch.nn.Module):
    """ SGC based on pytorch geometric. Simplifying Graph Convolutional Networks.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nclass : int
        size of output dimension
    K: int
        number of propagation in SGC
    cached : bool
        whether to set the cache flag in SGConv
    lr : float
        learning rate for SGC
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in SGC weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train SGC.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import SGC
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> sgc = SGC(nfeat=features.shape[1], K=3, lr=0.1,
              nclass=labels.max().item() + 1, device='cuda')
    >>> sgc = sgc.to('cuda')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> sgc.fit(pyg_data, train_iters=200, patience=200, verbose=True) # train with earlystopping
    """


    def __init__(self, nfeat, nclass, K=3, cached=True, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(SGC, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = SGConv(nfeat,
                nclass, bias=with_bias, K=K, cached=cached)

        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.name = "SGC"

    def forward(self, x, edge_index, edge_weight = None):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of SGC.
        """
        self.conv1.reset_parameters()

    def fit(self, pyg_data, train_iters=200, initialize=True, verbose=False, patience=500, **kwargs):
        """Train the SGC model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        # self.device = self.conv1.weight.device
        if initialize:
            self.initialize()

        self.data = pyg_data.to(self.device)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training SGC model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data.x, self.data.edge_index)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data.x, self.data.edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def predict(self, x ,edge_index, edge_weight = None):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of SGC
        """

        self.eval()
        return self.forward(x ,edge_index)






