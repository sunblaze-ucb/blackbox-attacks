#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=1
python generate_test_cwli.py test_thin_cwli_12.npy log 12 6705
