#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=5
python generate_test_cwli.py test_wide_cwli_8.npy log_wide 8 6705
