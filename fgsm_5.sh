#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=3
python generate_test_fgsm.py test_thin_fgsm_12.npy log 12
python generate_test_fgsm.py test_wide_fgsm_12.npy log_wide 12
python generate_test_fgsm.py test_tutorial_fgsm_12.npy tutorial/train 12

python generate_test_fgsm.py test_thin_fgsm_16.npy log 16
python generate_test_fgsm.py test_wide_fgsm_16.npy log_wide 16
python generate_test_fgsm.py test_tutorial_fgsm_16.npy tutorial/train 16
