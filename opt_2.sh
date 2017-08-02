#!/bin/sh -ex

# actually, just run each of these lines in screens for parallelism

# . ../venv/bin/activate
generate_seven() { # src ckpt_dir eps offset
	CUDA_VISIBLE_DEVICES=0 screen -S 0$1$3 -d -m python generate_test_optq.py queries/test_$1_opt_$3_0.npy $2 $3 $4
	CUDA_VISIBLE_DEVICES=1 screen -S 1$1$3 -d -m python generate_test_optq.py queries/test_$1_opt_$3_1.npy $2 $3 $4
	CUDA_VISIBLE_DEVICES=2 screen -S 2$1$3 -d -m python generate_test_optq.py queries/test_$1_opt_$3_2.npy $2 $3 $4
	CUDA_VISIBLE_DEVICES=3 screen -S 3$1$3 -d -m python generate_test_optq.py queries/test_$1_opt_$3_3.npy $2 $3 $4
	CUDA_VISIBLE_DEVICES=4 screen -S 4$1$3 -d -m python generate_test_optq.py queries/test_$1_opt_$3_4.npy $2 $3 $4
	CUDA_VISIBLE_DEVICES=5 screen -S 5$1$3 -d -m python generate_test_optq.py queries/test_$1_opt_$3_5.npy $2 $3 $4
	CUDA_VISIBLE_DEVICES=6 screen -S 6$1$3 -d -m python generate_test_optq.py queries/test_$1_opt_$3_6.npy $2 $3 $4
}

# generate_seven thin log 20 6705
# generate_seven thin log 24 6705
generate_seven thin log 28 6705

# generate_seven wide log_wide 20 6705
# generate_seven wide log_wide 24 6705
# generate_seven wide log_wide 28 6705

# generate_seven tutorial tutorial/train 20 6705
# generate_seven tutorial tutorial/train 24 6705
# generate_seven tutorial tutorial/train 28 6705

screen -RR
