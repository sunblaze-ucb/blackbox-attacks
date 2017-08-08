#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=5

python cifar10_query_based.py thin_32 --delta=1.0 --loss_type=xent
# python cifar10_query_based.py thin_32 --loss_type=cw
python cifar10_query_based.py thin_32_adv --delta=1.0 --loss_type=xent
# python cifar10_query_based.py thin_32_adv --loss_type=cw
python cifar10_query_based.py thin_32_ensadv --delta=1.0 --loss_type=xent
# python cifar10_query_based.py thin_32_ensadv --loss_type=cw
