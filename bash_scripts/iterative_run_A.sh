#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=0

python query_based_attack.py models/modelA --method=query_based_un_iter --loss_type=xent --delta=1.0
python query_based_attack.py models/modelA --method=query_based_un_iter --loss_type=cw