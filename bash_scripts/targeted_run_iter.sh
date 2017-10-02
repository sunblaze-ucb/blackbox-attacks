#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=0
python query_based_attack.py models/modelB --method=query_based_iter --loss_type=xent &

export CUDA_VISIBLE_DEVICES=3
python query_based_attack.py models/modelB --method=query_based_iter --loss_type=cw &

export CUDA_VISIBLE_DEVICES=4
python query_based_attack.py models/modelC --method=query_based_iter --loss_type=xent &
#
export CUDA_VISIBLE_DEVICES=5
python query_based_attack.py models/modelC --method=query_based_iter --loss_type=cw &
