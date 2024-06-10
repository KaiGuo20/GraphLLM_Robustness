python3 pgd_attack_baseline.py --model_name GCN  --seed_num 5 --dataset pubmed --split fixed --data_format llama_ft --ptb_rate 0.05 --attack PGD --hidden 256 --lr 0.001
python3 prbcd_attack_baseline.py --seed_num 5 --dataset arxiv --split fixed --ensemble_string llama\;know_exp_e5 --ptb_rate 0.2 --lr 0.001 --hidden 256




