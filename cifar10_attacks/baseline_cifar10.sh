#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=3

# python baseline_cifar10.py thin_32_adv
# python baseline_cifar10.py thin_32_ensadv
# python baseline_cifar10.py wide_28_10
# python baseline_cifar10.py wide_28_10_adv
# python baseline_cifar10.py wide_28_10_ensadv
python baseline_cifar10.py tutorial
python baseline_cifar10.py tutorial_adv
python baseline_cifar10.py tutorial_ensadv
