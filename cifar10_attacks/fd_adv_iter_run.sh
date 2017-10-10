#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=0

python cifar10_query_based.py thin_32_adv --method=query_based_un_iter --loss_type=xent --delta=1.0 &

# export CUDA_VISIBLE_DEVICES=1

# python cifar10_query_based.py thin_32_adv --method=query_based_un_iter --loss_type=cw &

export CUDA_VISIBLE_DEVICES=2

python cifar10_query_based.py thin_32_ensadv --method=query_based_un_iter --loss_type=xent --delta=1.0 &

# export CUDA_VISIBLE_DEVICES=4

# python cifar10_query_based.py thin_32_ensadv --method=query_based_un_iter --loss_type=cw &