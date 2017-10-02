#!/bin/sh -ex

# export CUDA_VISIBLE_DEVICES=0

# python query_based_attack.py models/modelA_adv --method=query_based_un_iter --loss_type=xent --delta=1.0 &

# export CUDA_VISIBLE_DEVICES=1

# python query_based_attack.py models/modelA_adv --method=query_based_un_iter --loss_type=cw &

# export CUDA_VISIBLE_DEVICES=2

# python query_based_attack.py models/modelA_ens_0.3_linf_ACD --method=query_based_un_iter --loss_type=xent --delta=1.0 &

# export CUDA_VISIBLE_DEVICES=3

# python query_based_attack.py models/modelA_ens_0.3_linf_ACD --method=query_based_un_iter --loss_type=cw &

# export CUDA_VISIBLE_DEVICES=4

# python query_based_attack.py models/modelA_ens_0.3_linf_ACD --method=query_based_un_iter --loss_type=xent --delta=1.0 &

# export CUDA_VISIBLE_DEVICES=7

# python query_based_attack.py models/modelA_ens_0.3_linf_ACD --method=query_based_un_iter --loss_type=cw &

export CUDA_VISIBLE_DEVICES=0

python query_based_attack.py models/modelA_adv --method=query_based_un_iter --loss_type=xent &

export CUDA_VISIBLE_DEVICES=2

python query_based_attack.py models/modelA_ens_0.3_linf_ACD --method=query_based_un_iter --loss_type=xent &

export CUDA_VISIBLE_DEVICES=4

python query_based_attack.py models/modelA_adv_0.3_linf_iter --method=query_based_un_iter --loss_type=xent &

export CUDA_VISIBLE_DEVICES=7

python query_based_attack.py models/modelA_adv_0.3_linf_iter --method=query_based_un_iter --loss_type=cw &

