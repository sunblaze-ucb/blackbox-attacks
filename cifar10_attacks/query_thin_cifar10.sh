#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=3

python cifar10_query_based.py thin_32 --loss_type=xent
python cifar10_query_based.py thin_32 --loss_type=cw
python cifar10_query_based.py thin_32_adv --loss_type=xent
python cifar10_query_based.py thin_32_adv --loss_type=cw
python cifar10_query_based.py thin_32_ensadv --loss_type=xent
python cifar10_query_based.py thin_32_ensadv --loss_type=cw