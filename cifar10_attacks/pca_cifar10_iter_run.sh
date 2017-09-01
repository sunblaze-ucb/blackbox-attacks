#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=4

python cifar10_query_based.py thin_32 --method=query_based_un_iter --loss_type=cw --num_comp=400 &

export CUDA_VISIBLE_DEVICES=5

python cifar10_query_based.py thin_32 --method=query_based_iter --loss_type=cw --num_comp=400 &

export CUDA_VISIBLE_DEVICES=6

python cifar10_query_based.py thin_32_adv --method=query_based_un_iter --loss_type=cw --num_comp=400 &

export CUDA_VISIBLE_DEVICES=7

python cifar10_query_based.py thin_32_adv --method=query_based_iter --loss_type=cw --num_comp=400 &