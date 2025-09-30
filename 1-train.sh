export PORT='10033'
CONFIG='configs/nuscenes/IAL_nusc_train_4xb2.py'
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh $CONFIG 4
