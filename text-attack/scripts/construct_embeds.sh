#!/bin/bash 
for data in  cora citeseer2 wikics pubmed arxiv history 
do
    for model in bow sbert e5 
    do
        python construct_embedding.py --dataset $data --model $model
    done
done


for data in  cora citeseer2 wikics pubmed arxiv history 
do
    python construct_embedding.py --dataset $data --model llama --finetune True
    python construct_embedding.py --dataset $data --model llama
done
