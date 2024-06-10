#!/bin/bash
# for model in sbert
# do
#     for dataset in cora citeseer2 pubmed wikics history
#     do
#         for victim in mlp gcn
#         do
#             for type in fixed
#             do
#                 python analysis.py --model $model --dataset $dataset --data_type $type --victim $victim
#             done
#         done
#     done
# done

# for model in llama
# do
#     for dataset in cora citeseer2 pubmed wikics history
#     do
#         for victim in mlp gcn
#         do
#             for type in fixed finetuned
#             do
#                 python analysis.py --model $model --dataset $dataset --data_type $type --victim $victim
#             done
#         done
#     done
# done


for model in sbert
do
    for dataset in pubmed wikics
    do
        for victim in gcn
        do
            for type in fixed 
            do
                python analysis.py --model $model --dataset $dataset --data_type $type --victim $victim
            done
        done
    done
done