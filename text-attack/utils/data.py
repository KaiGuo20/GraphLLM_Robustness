import sys
sys.path.append('')
import joblib
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy
from utils.util import get_dict

class NewDataset(Dataset):
    def __init__(self, args, path, model, tokenizer, device):
        self.max_len = 512
        self.num_candidates = 50
        self.save_path = path.split(".pkl")[0]+"_preprocessed.pkl"
        self.num_class = model.num_class

        if os.path.exists(self.save_path):
            self.data = joblib.load(self.save_path)
            self.ori_data = deepcopy(self.data)
        else:
            self.data = joblib.load(path)
            self.ori_data = deepcopy(self.data)

            self.seqs = []
            self.candidate_list = []
            self.new_embeddings = torch.FloatTensor([]).to(device)
            
            correct = 0
            for item in tqdm(self.data):
                self.seqs.append(item['seq'])
                # item['seq'] = torch.nn.functional.pad(torch.LongTensor(item['seq']),(0, self.max_len - item['seq_len']), 'constant', 0)
                # item['seq_embedding'] = model.model.embeddings.word_embeddings(torch.LongTensor(item['seq']).to(device)).cpu()
                # cw_mask = torch.zeros(item['seq_embedding'].shape).float()
                # cw_mask[1:item['seq_len']] = 1
                # item['mask'] = cw_mask
                embed = model.generate(input_ids = torch.LongTensor([item['seq']]), seq_len = torch.LongTensor([item['seq_len']]))
                self.new_embeddings = torch.cat([self.new_embeddings, embed], dim = 0)

                if args.function == 'all':
                    pass
                elif args.function == 'knowledge':
                    all_dict = get_dict(item['knowledge_dict'], tokenizer)
                elif args.function == 'typo':
                    all_dict = get_dict(item['bug_dict'], tokenizer)
                else:
                    raise Exception('Unknown perturbation function.')
                
                self.candidate_list.append(all_dict)

            #     candidates = torch.LongTensor([])
            #     candidates_embeds = torch.LongTensor([])
            #     for i in range(1, item['seq_len']):
            #         candidate = torch.LongTensor(all_dict[item['seq'][i].item()])
            #         candidate_embed = model.model.embeddings.word_embeddings(candidate.to(device)).cpu()[:self.num_candidates]
            #         empty = self.num_candidates - candidate_embed.shape[0]
            #         if empty > 0:
            #             candidate_embed = torch.cat([candidate_embed, candidate_embed[0].unsqueeze(0).repeat(empty, 1)]).unsqueeze(0)
            #             candidate = torch.cat([candidate, candidate[0].unsqueeze(0).repeat(empty)]).unsqueeze(0)
            #         else:
            #             candidate_embed = candidate_embed[:self.num_candidates].unsqueeze(0)
            #             candidate = candidate[:self.num_candidates].unsqueeze(0)

            #         candidates_embeds = torch.concat([candidates_embeds, candidate_embed])
            #         candidates = torch.concat([candidates, candidate])

            #     item['all_dict_embedding'] = torch.nn.functional.pad(candidates_embeds, (0, 0, 0, 0, 0, self.max_len - item['seq_len']), 'constant', 0)
            #     item['all_dict'] = torch.nn.functional.pad(candidates, (0, 0, 0, self.max_len - item['seq_len']), 'constant', 0)
            #     # test original acc
            #     out = model(input_ids = item['seq'].unsqueeze(0), seq_len = torch.LongTensor([item['seq_len']]))['pred']
            #     ori_prediction = torch.max(out, 1)[1].item()

            #     correct += int(ori_prediction == item['label'])
            # print(f"preprocess finished, original acc: {correct/len(self.data)}")
        
        # for item in tqdm(self.data):
        #     del item['Node']
        #     del item['raw_text']
        #     del item['category_name']
        #     del item['knowledge_dict']

        print(f"preprocess finished, preprocessed dataset loaded")


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        data = self.data[index]
        return index, data
    

