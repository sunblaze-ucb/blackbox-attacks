#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=1
python query_based_attack.py models/modelD --method=query_based_iter --loss_type=xent &

export CUDA_VISIBLE_DEVICES=2
python query_based_attack.py models/modelD --method=query_based_iter --loss_type=cw &

# export CUDA_VISIBLE_DEVICES=4
# python query_based_attack.py models/modelC --method=query_based --loss_type=xent &
#
# export CUDA_VISIBLE_DEVICES=5
# python query_based_attack.py models/modelC --method=query_based --loss_type=cw &
