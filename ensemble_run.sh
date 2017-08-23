#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=6

python ensemble_attack.py fgs models/modelB --target_model=models/modelA --loss_type=cw
python ensemble_attack.py fgs models/modelB --target_model=models/modelA_adv --loss_type=cw
python ensemble_attack.py fgs models/modelB --target_model=models/modelA_ens_0.3_linf_ACD --loss_type=cw
python ensemble_attack.py fgs models/modelB --target_model=models/modelA_adv_0.3_linf_iter --loss_type=cw

python ensemble_attack.py fgs models/modelB models/modelC --target_model=models/modelA --loss_type=xent
python ensemble_attack.py fgs models/modelB models/modelC --target_model=models/modelA_adv --loss_type=xent
python ensemble_attack.py fgs models/modelB models/modelC --target_model=models/modelA_ens_0.3_linf_ACD --loss_type=xent
python ensemble_attack.py fgs models/modelB models/modelC --target_model=models/modelA_adv_0.3_linf_iter --loss_type=xent

python ensemble_attack.py fgs models/modelB models/modelC --target_model=models/modelA --loss_type=cw
python ensemble_attack.py fgs models/modelB models/modelC --target_model=models/modelA_adv --loss_type=cw
python ensemble_attack.py fgs models/modelB models/modelC --target_model=models/modelA_ens_0.3_linf_ACD --loss_type=cw
python ensemble_attack.py fgs models/modelB models/modelC --target_model=models/modelA_adv_0.3_linf_iter --loss_type=cw
