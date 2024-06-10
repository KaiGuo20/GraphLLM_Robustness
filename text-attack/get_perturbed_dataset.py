import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from attacks.models import *
import attacks.models as models
from copy import deepcopy
from attacks.pyg_defence import GCN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora")
parser.add_argument('--data_type', type=str, default="fixed") # fixed finetuned
parser.add_argument('--model', type=str, default='sbert')
parser.add_argument('--victim', type=str, default='mlp')
parser.add_argument('--sample_choice', type=str, default="random")
parser.add_argument('--edge_ptb', type=float, default=0)
parser.add_argument('--node_ptb', type=float, default=0.05)
args = parser.parse_args()

device = torch.device("cuda")

# model = models.BertC(dropout=0.5, num_class=3, device = device).to(device)
# model.eval()
data_name = args.dataset

edge_ptb = args.edge_ptb
node_ptb = args.node_ptb
sample = args.sample_choice

setting = f"edge_{edge_ptb}_node_{node_ptb}_{sample}"

tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.dataset != 'arxiv':
    ori_data_path= f"raw_data/{args.dataset}_{args.data_type}_{args.model}.pt"
    ori_data = torch.load(ori_data_path, map_location = 'cpu')
    del ori_data.seqs
    del ori_data.candidates
    print(ori_data)

# ori_data.x = get_sbert_embedding(ori_data.raw_texts)

new_texts_list = []
new_edge_index_list = []
new_edge_weight_list = []
new_features = []

ori_txt = []
ptb_txt = []

mod = []
mod_seq = []
mod_nodes_list = []
mod_result = []
for split in range(5):

    if args.dataset == 'arxiv':
        ori_data_path= f"raw_data/{args.dataset}_{args.data_type}_{args.model}_{split}.pt"
        ori_data = torch.load(ori_data_path, map_location = 'cpu')
        del ori_data.seqs
        del ori_data.candidates
        print(ori_data)

    
    if args.data_type == 'finetuned':
        ptb_data_path = f"{args.victim}_results/{data_name}_{args.model}/{setting}/finetuned_split_{split}.pt"
    else:
        ptb_data_path = f"{args.victim}_results/{data_name}_{args.model}/{setting}/split_{split}.pt"

    ptb_data = torch.load(ptb_data_path)
    mod_nodes_list.append(ptb_data.mod_nodes)
    try:
        new_texts = ptb_data.raw_text
    except:
        new_texts = ptb_data.raw_texts
    #-----get attack result
    print(f"running split {split}")
    ori_data.train_mask = ori_data.train_masks[split].cpu()
    ori_data.val_mask = ori_data.val_masks[split].cpu()
    ori_data.test_mask = ori_data.test_masks[split].cpu()
    ori_data.train_mask = torch.where(ori_data.train_mask == True)[0].numpy()
    ori_data.val_mask = torch.where(ori_data.val_mask == True)[0].numpy()
    ori_data.test_mask = torch.where(ori_data.test_mask == True)[0].numpy()
    labels = ori_data.y.to(device)
    def get_eval(dataset, model_name = 'mlp'):
        if model_name == 'mlp':
            model = new_MLP(nfeat=ori_data.x.shape[1], nhid=256, dropout=0.5,
                        nlayers=2, with_bn=False, weight_decay=5e-4, nclass=labels.max().item() + 1,
                        device=device).to(device)
        elif model_name == 'gcn':
            model = GCN(nfeat=ori_data.x.shape[1], nhid=256, dropout=0.5,
                            nlayers=2, with_bn=False, weight_decay=5e-4, nclass=labels.max().item() + 1, lr = 0.01,
                            device=device).to(device)

        model.fit(ori_data.to(device), train_iters=1000, patience=200, verbose=True) # iter: 1000
        model.eval()
        # model.data = data.to(self.device)
        # ipdb.set_trace()
        output = model.predict(ori_data.x.to(device), ori_data.edge_index.to(device))
        clean_pred = output.argmax(1)

        mod_nodes = dataset.mod_nodes

        ptb_out = model.predict(dataset.x.to(device), ori_data.edge_index.to(device))
        ptb_pred = ptb_out.argmax(1)

        clean_pred = clean_pred[mod_nodes]
        ptb_pred = ptb_pred[mod_nodes]
        # ipdb.set_trace()
        ptb_result = (clean_pred == ptb_pred).tolist()
        clean_result = (clean_pred == labels[mod_nodes]).tolist()

        return dict(zip(mod_nodes.tolist(), ptb_result)), dict(zip(mod_nodes.tolist(), clean_result))

    result = get_eval(ptb_data, model_name = args.victim)
    mod_result.append(result)

    
    #-----get ptb_texts
    if node_ptb != 0:
        new_texts = []
        cur_ptb_txt = []
        try:
            ori_text = ori_data.raw_text
        except:
            ori_text = ori_data.raw_texts
        for idx, txt in enumerate(ori_text):
            if idx in ptb_data.mod_nodes:
                i = list(ptb_data.mod_nodes.numpy()).index(idx)
                seq = ptb_data.perturbed_seqs[i]
                seq = seq[torch.where(seq!=0)]

                mod_seq.append(seq)

                seq = seq[1:-1]
                new_text = tokenizer.decode(seq)

                new_texts.append(new_text)
                ptb_txt.append(new_text)
                cur_ptb_txt.append(new_text)
                ori_txt.append(txt) 
                mod.append(idx)
            else:
                new_texts.append(txt)
    
    
    # ptb_embeds = torch.FloatTensor([]).to(device)
    # for idx, txt in enumerate(cur_ptb_txt):
    #     feat = model.generate(txt = txt)
    #     ptb_embeds = torch.cat([ptb_embeds, feat])
    # ptb_embeds = ptb_embeds.detach().cpu()

    # new_x = deepcopy(ori_data.x)
    # new_x[ptb_data.mod_nodes] = ptb_embeds
    new_features.append(ptb_data.x.cpu())
    new_edge_index_list.append(ptb_data.edge_index)
    new_edge_weight_list.append(ptb_data.edge_weight)
    new_texts_list.append(new_texts)

ori_data.mod_result = mod_result
ori_data.ptb_texts = new_texts_list
ori_data.mod_nodes = mod_nodes_list
ori_data.ptb_features = new_features
ori_data.ptb_edge_index = new_edge_index_list
ori_data.ptb_edge_weight = new_edge_weight_list


# print(len(ori_data.ptb_texts[0]))

# print(mod[12])
# print(mod_seq[11])
for i in range(5):
    print(i,":")
    print(ptb_txt[i])
    print("\n")
    print(ori_txt[i])
torch.save(ori_data, f"processed_data/ptb_datasets/{args.victim}_{data_name}_{args.data_type}_{args.model}_{args.node_ptb}.pt")
print("done!")




