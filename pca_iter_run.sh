#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=0
python query_based_attack.py models/modelA --method=query_based_un_iter --loss_type=cw --num_comp=400 &

export CUDA_VISIBLE_DEVICES=1
python query_based_attack.py models/modelA --method=query_based_iter --loss_type=cw --num_comp=400 &

export CUDA_VISIBLE_DEVICES=2
python query_based_attack.py models/modelA_adv --method=query_based_un_iter --loss_type=cw --num_comp=400 &

export CUDA_VISIBLE_DEVICES=3
python query_based_attack.py models/modelA_adv --method=query_based_iter --loss_type=cw --num_comp=400 &
