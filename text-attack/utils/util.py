import errno
import json
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import pickle
import re
import jieba

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default='knowledge',
                        choices=['all', 'typo', 'knowledge', 'cluster'],
                        help='purturbation function')
                        
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--data_type", type=str, default="fixed")
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--ptb", default = "FK")


    
    parser.add_argument('--train-data', type=str, default='processed_data/Cora/train_data.pkl',
                        help='location of the training data, should be a json file')
    parser.add_argument('--test-data', type=str, default='processed_data/Cora/data_sbert_FK.pkl',
                        help='location of the test data, should be a json file')
    parser.add_argument('--embedding-data', type=str, default='raw_data/s.pt',
                        help='location of the embedding data, should be a json file')
    parser.add_argument('--word-list', type=str, default='raw_data/word_list.pkl',
                        help='location of the word list data, should be a json file')
    

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--lr', type=float, default=.1,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for training')
    parser.add_argument('--const', type=float, default=1e4,
                        help='initial const for cw attack')
    parser.add_argument('--confidence', type=float, default=0,
                        help='initial const for cw attack')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='cw max steps')
    parser.add_argument('--l1', action='store_true',
                        help='use l1 norm')
    return parser.parse_args()

def difference(a, b):
    tot = 0
    for x, y in zip(a, b):
        if x != y:
            tot += 1
    return tot

def get_dict(similar_dict, tokenizer):
    new_dict = {0: [0], 101: [101]}
    for k, v in similar_dict.items():
        k = tokenizer.convert_tokens_to_ids(k)
        v = [tokenizer.convert_tokens_to_ids(x) for x in v]
        if k not in v:
            v.append(k)
        while 100 in v:
            v.remove(100)
        if len(v) >= 1:
            new_dict[k] = v
        else:
            new_dict[k] = [k]

    return new_dict

def token_to_text(seq, tokenizer):
    return [tokenizer.decode(tokens[1:-1]) for tokens in seq]

def transform(seq, tokenizer, unk_words_dict=None):
    if unk_words_dict is None:
        unk_words_dict = {}
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    unk_count = 0
    for x in seq:
        if x == 100:
            unk_count += 1
    if unk_count == 0 or len(unk_words_dict) == 0:
        return tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(x) for x in seq])
    else:
        tokens = []
        for idx, x in enumerate(seq):
            if x == 100 and len(unk_words_dict[idx]) != 0:
                unk_words = unk_words_dict[idx]
                unk_word = random.choice(unk_words)
                tokens.append(unk_word)
            else:
                tokens.append(tokenizer.convert_ids_to_tokens(x))
        return tokenizer.convert_tokens_to_string(tokens)

def has_letter(word):
    """Returns true if `word` contains at least one character in [A-Za-z]."""
    return re.search("[A-Za-z]+", word) is not None

def words_from_text(s, words_to_ignore=[]):
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    try:
        if re.search("[\u4e00-\u9FFF]", s):
            seg_list = jieba.cut(s, cut_all=False)
            s = " ".join(seg_list)
        else:
            s = " ".join(s.split())
    except Exception:
        s = " ".join(s.split())

    homos = """˗৭Ȣ𝟕бƼᏎƷᒿlO`ɑЬϲԁе𝚏ɡհіϳ𝒌ⅼｍոорԛⲅѕ𝚝սѵԝ×уᴢ"""
    exceptions = """'-_*@"""
    filter_pattern = homos + """'\\-_\\*@"""
    # TODO: consider whether one should add "." to `exceptions` (and "\." to `filter_pattern`)
    # example "My email address is xxx@yyy.com"
    filter_pattern = f"[\\w{filter_pattern}]+"
    words = []
    for word in s.split():
        # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the word.
        word = word.lstrip(exceptions)
        filt = [w.lstrip(exceptions) for w in re.findall(filter_pattern, word)]
        words.extend(filt)
    words = list(filter(lambda w: w not in words_to_ignore + [""], words))
    return words


def init_logger():
    if not os.path.exists(root_dir):
        os.mkdir('./'+root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='a')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    
            
def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
def write_pickle(filename, file):
    with open(filename, "wb") as f:
        return pickle.dump(file, f)
def load_file(filename):
    with open(filename, "rb") as f:
        return f.readlines()




args = get_args()
args.untargeted = True
root_dir = os.path.join('./results', args.function, 'untargeted')

make_sure_path_exists(root_dir)

logger = init_logger()

PAD = 0
UNK = 1
# not part of the qa vocab, assigned with minus index
EOS = -1
SOS = -2

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
EOS_WORD = '<eos>'
SOS_WORD = '<sos>'
