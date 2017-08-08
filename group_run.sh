#!/bin/sh -ex

export CUDA_VISIBLE_DEVICES=6

# python query_based_attack.py models/modelA --loss_type=cw --group_size=7
# python query_based_attack.py models/modelA --loss_type=cw --group_size=35
# python query_based_attack.py models/modelA --loss_type=cw --group_size=70
# python query_based_attack.py models/modelA --loss_type=cw --group_size=140
# python query_based_attack.py models/modelA --loss_type=cw --group_size=350
#
# python query_based_attack.py models/modelA_adv --loss_type=cw --group_size=7
# python query_based_attack.py models/modelA_adv --loss_type=cw --group_size=35
# python query_based_attack.py models/modelA_adv --loss_type=cw --group_size=70
# python query_based_attack.py models/modelA_adv --loss_type=cw --group_size=140
# python query_based_attack.py models/modelA_adv --loss_type=cw --group_size=350

python query_based_attack.py models/modelA --loss_type=cw --num_comp=10
python query_based_attack.py models/modelA --loss_type=cw --num_comp=50
python query_based_attack.py models/modelA --loss_type=cw --num_comp=100
python query_based_attack.py models/modelA --loss_type=cw --num_comp=200
python query_based_attack.py models/modelA --loss_type=cw --num_comp=400

python query_based_attack.py models/modelA_adv --loss_type=cw --num_comp=10
python query_based_attack.py models/modelA_adv --loss_type=cw --num_comp=50
python query_based_attack.py models/modelA_adv --loss_type=cw --num_comp=100
python query_based_attack.py models/modelA_adv --loss_type=cw --num_comp=200
python query_based_attack.py models/modelA_adv --loss_type=cw --num_comp=400
