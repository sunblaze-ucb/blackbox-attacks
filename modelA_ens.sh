#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=5

python query_based_attack.py models/modelA_ens_0.3_linf_ACD --delta=1.0 --loss_type=xent --alpha=0.05
python query_based_attack.py models/modelA_ens_0.3_linf_ACD --delta=1.0 --loss_type=cw --alpha=0.05