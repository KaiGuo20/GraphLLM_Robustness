#!/bin/bash
for model in llama
do
    for dataset in arxiv # cora citeseer2 pubmed wikics history
    do
        for victim in gcn
        do
            for type in fixed finetuned
            do
                python get_perturbed_dataset.py --model $model --dataset $dataset --victim $victim --data_type $type
            done
        done
    done
done