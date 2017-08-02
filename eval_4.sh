#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=7

predict() { # src dst ckpt_dir
	python predict_file.py queries/preds_$1_$2_0.npy queries/test_$1_0.npy $3
	python predict_file.py queries/preds_$1_$2_1.npy queries/test_$1_1.npy $3
	python predict_file.py queries/preds_$1_$2_2.npy queries/test_$1_2.npy $3
	python predict_file.py queries/preds_$1_$2_3.npy queries/test_$1_3.npy $3
	python predict_file.py queries/preds_$1_$2_4.npy queries/test_$1_4.npy $3
	python predict_file.py queries/preds_$1_$2_5.npy queries/test_$1_5.npy $3
	python predict_file.py queries/preds_$1_$2_6.npy queries/test_$1_6.npy $3
}


predict_and_compare() { # src dst ckpt_dir offset
	predict $1 $2 $3
	python compare_gt.py preds_orig_$2.npy $4 \
		queries/preds_$1_$2_0.npy \
		queries/preds_$1_$2_1.npy \
		queries/preds_$1_$2_2.npy \
		queries/preds_$1_$2_3.npy \
		queries/preds_$1_$2_4.npy \
		queries/preds_$1_$2_5.npy \
		queries/preds_$1_$2_6.npy
}

# opt 20-24-28

# predict_and_compare tutorial_opt_20 wide log_wide 6705
# predict_and_compare tutorial_opt_24 wide log_wide 6705
# predict_and_compare tutorial_opt_28 wide log_wide 6705
# predict_and_compare tutorial_opt_20 wide_adv log_adv_wide 6705
# predict_and_compare tutorial_opt_24 wide_adv log_adv_wide 6705
# predict_and_compare tutorial_opt_28 wide_adv log_adv_wide 6705
# predict_and_compare tutorial_opt_20 wide_ensadv log_ensadv_wide 6705
# predict_and_compare tutorial_opt_24 wide_ensadv log_ensadv_wide 6705
# predict_and_compare tutorial_opt_28 wide_ensadv log_ensadv_wide 6705

# predict_and_compare thin_opt_20 tutorial tutorial/train 6705
# predict_and_compare thin_opt_24 tutorial tutorial/train 6705
predict_and_compare thin_opt_28 tutorial tutorial/train 6705
# predict_and_compare thin_opt_20 tutorial_adv tutorial/train_adv 6705
# predict_and_compare thin_opt_24 tutorial_adv tutorial/train_adv 6705
predict_and_compare thin_opt_28 tutorial_adv tutorial/train_adv 6705
# predict_and_compare thin_opt_20 tutorial_ensadv tutorial/train_ensadv 6705
# predict_and_compare thin_opt_24 tutorial_ensadv tutorial/train_ensadv 6705
predict_and_compare thin_opt_28 tutorial_ensadv tutorial/train_ensadv 6705

# predict_and_compare wide_opt_20 thin log 6705
# predict_and_compare wide_opt_24 thin log 6705
# predict_and_compare wide_opt_28 thin log 6705
# predict_and_compare wide_opt_20 thin_adv log_adv 6705
# predict_and_compare wide_opt_24 thin_adv log_adv 6705
# predict_and_compare wide_opt_28 thin_adv log_adv 6705
# predict_and_compare wide_opt_20 thin_ensadv log_ensadv 6705
# predict_and_compare wide_opt_24 thin_ensadv log_ensadv 6705
# predict_and_compare wide_opt_28 thin_ensadv log_ensadv 6705
