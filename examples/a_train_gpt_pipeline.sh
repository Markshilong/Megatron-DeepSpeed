#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


CHECKPOINT_PATH=checkpoints/gpt2-pipeline
VOCAB_FILE=data/gpt2-vocab.json
MERGE_FILE=data/gpt2-merges.txt
DATA_PATH=data/meg-gpt2-oscar-en-10k_text_document
TENSORBOARD_PATH=output_dir/tensorboard

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
       --nnodes $NNODES \
       --node_rank $NODE_RANK \
       --master_addr $MASTER_ADDR \
       --master_port $MASTER_PORT"

# remove last checkpoints
rm -rf $CHECKPOINT_PATH

# ZERO_STAGE=3
# MICRO_BATCH_SIZE=4
# GLOBAL_BATCH_SIZE=16

# config_json="./ds_config.json"

# # Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
# cat <<EOT > $config_json
# {
#   "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
#   "train_batch_size": $GLOBAL_BATCH_SIZE,
#   "gradient_clipping": 1.0,
#   "zero_optimization": {
#     "stage": $ZERO_STAGE,
#     "offload_param": {
#       "device": "nvme",
#       "nvme_path": "/home/shilonglei/OOC/nvme_offload",
#       "pin_memory": true,
#       "buffer_count": 6,
#       "buffer_size": 1e8,
#       "max_in_cpu": 1e9
#     },
#     "overlap_comm": true,
#     "contiguous_gradients": true,
#     "reduce_bucket_size": 1048576,
#     "stage3_prefetch_bucket_size": 104858,
#     "stage3_max_live_parameters": 1e8,
#     "stage3_max_reuse_distance": 1e8,
#     "stage3_param_persistence_threshold": 10240
#   },
#   "aio": {
#     "block_size": 131072,
#     "queue_depth": 16,
#     "thread_count": 1,
#     "single_submit": true,
#     "overlap_events": true
#   },
#   "fp16": {
#     "enabled": true,
#     "loss_scale": 0,
#     "loss_scale_window": 500,
#     "hysteresis": 2,
#     "min_loss_scale": 1,
#     "initial_scale_power": 12
#   },
#   "steps_per_print": 2000,
#   "wall_clock_breakdown": false
# }
# EOT

ZERO_STAGE=1
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       /home/shilonglei/OOC/Megatron-DeepSpeed/pretrain_gpt.py \
       $DEEPSPEED_ARGS \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 4 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 30 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 5 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
