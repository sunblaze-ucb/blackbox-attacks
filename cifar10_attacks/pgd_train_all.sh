#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=0

#python baseline_cifar10.py thin_32_pgd

#python baseline_cifar10.py thin_32_pgd --alpha=33

python cifar10_query_based.py thin_32_pgd --method=query_based_un_iter --loss_type=xent --delta=1.0 & 

export CUDA_VISIBLE_DEVICES=1

python cifar10_query_based.py thin_32_pgd --method=query_based_un_iter --loss_type=cw & 