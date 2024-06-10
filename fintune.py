import torch as th
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import os.path as osp
from utils import init_random_state, init_path, eval
from utils import compute_loss
from data import set_seed_config
import numpy as np
import torch.nn.functional as F
from transformers.models.auto import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer import Trainer, TrainingArguments, IntervalStrategy
import argparse
from torch_geometric.utils import mask_to_index
import ipdb
from ogb.nodeproppred import Evaluator
import os
from data import get_dataset
from utils import knowledge_augmentation
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from args import *
import wandb
wandb.login(key="8ae79e5dfdfb05a0b5ec9ae46781aad4ef68cd55")

args = get_command_line_args()
base_model_name = "meta-llama/Llama-2-7b-hf"
data = get_dataset(args.seed_num, args.dataset, args.split, args.data_format, args.low_label_test)
# print('data', data)
# print('y', len(data.category_names))
data.train_mask = data.train_masks[0]
data.val_mask = data.val_masks[0]
data.test_mask = data.test_masks[0]
############
from datasets import Dataset, ClassLabel, Features,Value

# 假设你有两个列表，一个是文本内容，另一个是标签
train_mask_list = data.train_mask.tolist()
train_texts = [data.raw_texts[i] for i in range(len(train_mask_list)) if train_mask_list[i]]

train_labels = [data.category_names[i] for i in range(len(train_mask_list)) if train_mask_list[i]]
train_NUM_CLASSES = len(set(train_labels))
train_CLASS_NAMES = list(set(train_labels))

val_mask_list = data.val_mask.tolist()
val_texts = [data.raw_texts[i] for i in range(len(val_mask_list)) if val_mask_list[i]]
val_labels = [data.category_names[i] for i in range(len(val_mask_list)) if val_mask_list[i]]
val_NUM_CLASSES = len(set(val_labels))
val_CLASS_NAMES = list(set(val_labels))
train_features = Features({
    'text': Value('string'),  # 定义文本特征为字符串类型
    'label': ClassLabel(names=list(set(train_labels)))  # 假设标签为这几种类别
})
# 创建数据字典
train_data_dict = {
    'text': train_texts,
    'label': train_labels
}
# 创建数据集对象
train_dataset = Dataset.from_dict(train_data_dict, features=train_features)
print(train_dataset)
print('dataset', train_dataset.features)
##########
val_features = Features({
    'text': Value('string'),  # 定义文本特征为字符串类型
    'label': ClassLabel(names=list(set(val_labels)))  # 假设标签为这几种类别
})
# 创建数据字典
val_data_dict = {
    'text': val_texts,
    'label': val_labels
}
# 创建数据集对象
val_dataset = Dataset.from_dict(val_data_dict, features=val_features)
print(val_dataset)
#####################################
def formatting_func(example):
    text = f"### Question: {example['text']}\n ### Answer: {example['label']}"
    return text
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = val_dataset.map(generate_and_tokenize_prompt)
    ##############this
base_model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# device_map = {"": 0}
device_map="auto"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    token='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',
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
)
base_model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

output_dir = "./results/history"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500
)

max_seq_length = 512

trainer = SFTTrainer(
    model=base_model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()


import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)