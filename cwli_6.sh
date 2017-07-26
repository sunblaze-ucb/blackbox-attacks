#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=3
python generate_test_cwli.py test_tutorial_cwli_12.npy tutorial/train 12 6705
