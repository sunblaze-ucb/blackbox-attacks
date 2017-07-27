#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=3
python generate_test_cwli.py test_wide_cwli_16.npy log_wide 16 6705
