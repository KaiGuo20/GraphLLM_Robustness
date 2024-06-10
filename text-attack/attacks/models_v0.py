import numpy as np
from torch import nn
import torch
from pytorch_transformers.modeling_bert import *
from sentence_transformers import SentenceTransformer
from pytorch_transformers import BertTokenizer, BertModel
from torch_geometric.nn.models import MLP
from tqdm import tqdm

def seq_len_to_mask(seq_len, max_len=None):
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, "seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        if max_len is None:
            max_len = int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, "seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        if max_len is None:
            max_len = seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

class BertEmbeddings_attack(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings_attack, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, perturbed=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if perturbed is not None:
            # print("output embedding:", perturbed)
            embeddings = perturbed + position_embeddings + token_type_embeddings
        else:
            # print("out embedding:", words_embeddings)
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel_attack(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel_attack, self).__init__(config)

        self.embeddings = BertEmbeddings_attack(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
                perturbed=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                           perturbed=perturbed)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertC(nn.Module):
    def __init__(self, name='bert-base-uncased', dropout=0.5, num_class=5, device = None):
        super(BertC, self).__init__()
        self.device = device
        self.max_len = 512
        config = BertConfig.from_pretrained(name)
        self.bert = BertModel_attack(config).from_pretrained(name)
        self.tokenizer = BertTokenizer.from_pretrained(name)
        for param in self.bert.parameters():
            param.requires_grad = False
        #self.proj = nn.Linear(config.hidden_size, num_class)
        #self.drop = nn.Dropout(p=dropout)

        self.loss_f = nn.CrossEntropyLoss()
        
        self.mlp = MLP(channel_list=[768, 256, num_class], dropout=dropout, norm=None)

    def forward(self, src, seq_len, gold=None, perturbed=None):
        src_mask = seq_len_to_mask(seq_len, src.size(1))
        out = self.bert(src, attention_mask=src_mask, perturbed=perturbed)
        embed = out[1]
        # print(embed.size())
        # logits = self.proj(self.drop(embed))
        logits = self.mlp(embed)
        ret = {"pred": logits}
        if gold is not None:
            ret["loss"] = self.loss_f(logits, gold)
        ret['embedding'] = out[0]
        return ret
    
    def generate(self, txt, perturbed = None):
        txt = ['[CLS] ' + t for t in txt]
        src = [self.tokenizer.encode(words) for words in txt]
        src = torch.LongTensor([toks if len(toks) <= self.max_len else toks[:self.max_len] for toks in src]).to(self.device)
        seq_lens = torch.LongTensor([len(toks) for toks in src]).to(self.device)
        src_mask = seq_len_to_mask(seq_lens, src.size(1))
        out = self.bert(src, attention_mask=src_mask.to(self.device), perturbed=perturbed)
        embed = out[1]
        return embed



def get_sbert_embedding(texts):
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/localscratch/czk/huggingface', device='cuda').to('cuda')
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return sbert_model, torch.tensor(sbert_embeds)

#-------------------mlp--------------------------------------

from torch_geometric.nn.models import MLP
import torch
import torch.nn.functional as F
from copy import deepcopy
import torch
from ogb.nodeproppred import Evaluator


class UniversalMLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None, return_embeds = False) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hidden_dimensions = [hidden_dimension] * (num_layers - 1)
        self.hidden_dimensions = [input_dim] + hidden_dimensions + [num_classes]
        self.dropout = dropout
        self.norm = norm
        self.mlp = MLP(channel_list=self.hidden_dimensions, dropout=self.dropout, norm=self.norm).to(self.device)
        self.return_embeds = False
    
    def reset_params(self):
        self.mlp = MLP(channel_list=self.hidden_dimensions, dropout=self.dropout, norm=self.norm).to(self.device)

    def forward(self, x):
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



