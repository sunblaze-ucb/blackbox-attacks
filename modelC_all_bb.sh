#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=6

# python baseline_attacks.py models/modelC
# python baseline_attacks.py models/modelC --alpha=0.51
# python query_based_attack.py models/modelC --delta=1.0 --loss_type=xent
python query_based_attack.py models/modelC --delta=0.01 --loss_type=cw
