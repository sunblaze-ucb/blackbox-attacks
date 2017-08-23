#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=2

python cifar10_query_based.py thin_32 --loss_type=cw --num_comp=10
python cifar10_query_based.py thin_32 --loss_type=cw --num_comp=50
python cifar10_query_based.py thin_32 --loss_type=cw --num_comp=100
python cifar10_query_based.py thin_32 --loss_type=cw --num_comp=200
python cifar10_query_based.py thin_32 --loss_type=cw --num_comp=400