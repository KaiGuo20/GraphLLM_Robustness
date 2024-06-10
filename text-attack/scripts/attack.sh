#!/bin/bash
for data in cora pubmed
do 
    for model in bow sbert e5
    do
        for edge_ptb in 0 0.05
        do
            for node_ptb in 0 0.05
            do
                python test_node_prbcd.py --dataset $data --edge_ptb_rate $edge_ptb --node_ptb_rate $node_ptb --model $model
                python test_node_prbcd.py --dataset $data --edge_ptb_rate $edge_ptb --node_ptb_rate $node_ptb --model $model
            done
        done
    done
done


