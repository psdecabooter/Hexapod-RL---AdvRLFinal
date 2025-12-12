#!/bin/bash
# Quick training script for hexapod with model saving enabled

python cleanrl/ppo_continuous_action.py \
    --env-id Hexapod-v0 \
    --total-timesteps 2000000 \
    --num-envs 8 \
    --save-model \
    --seed 564 \
    --exp-name forward_motion_hexapod \
    # --capture-video
