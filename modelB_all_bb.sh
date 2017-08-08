#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=4

python baseline_attacks.py models/modelB
python baseline_attacks.py models/modelB --alpha=0.51
python query_based_attack.py models/modelB --delta=1.0 --loss_type=xent
python query_based_attack.py models/modelB --delta=0.01 --loss_type=cw
