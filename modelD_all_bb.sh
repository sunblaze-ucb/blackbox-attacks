#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=7

# python baseline_attacks.py models/modelD
# python baseline_attacks.py models/modelD --alpha=0.51
# python query_based_attack.py models/modelD --delta=1.0 --loss_type=xent
python query_based_attack.py models/modelD --delta=0.01 --loss_type=cw
