#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=4

python query_based_attack.py models/modelA --method=query_based_iter --loss_type=xent --num_comp=50
python query_based_attack.py models/modelA --method=query_based_iter --loss_type=cw --num_comp=50

python query_based_attack.py models/modelA_adv --method=query_based_iter --loss_type=xent --num_comp=50
python query_based_attack.py models/modelA_adv --method=query_based_iter --loss_type=cw --num_comp=50
