#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=0

python cifar10_query_based.py tutorial --loss_type=xent
python cifar10_query_based.py tutorial --loss_type=cw
python cifar10_query_based.py tutorial_adv --loss_type=xent
python cifar10_query_based.py tutorial_adv --loss_type=cw
python cifar10_query_based.py tutorial_ensadv --loss_type=xent
python cifar10_query_based.py tutorial_ensadv --loss_type=cw