CONFIG_FILE='configs/nuscenes/IAL_nusc_train_4xb2.py'
CKPT_PATH='checkpoints/ial_nusc_val/epoch_83.pth'
GPUS=2
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29503}
MASTER_ADDR=10044

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=$NNODES --node_rank=0 --master_addr=127.10.29.1 --nproc_per_node=$GPUS --master_port=$MASTER_ADDR \
 scripts/test.py $CONFIG_FILE $CKPT_PATH --launcher pytorch
