#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=6
# python generate_test_fgsm.py test_thin_fgsm_8.npy log 8
# python generate_test_fgsm.py test_wide_fgsm_8.npy log_wide 8
python generate_test_fgsm.py test_tutorial_fgsm_8.npy tutorial/train 8
