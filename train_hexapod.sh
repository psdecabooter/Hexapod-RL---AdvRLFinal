#!/bin/bash
# Quick training script for hexapod with model saving enabled

python cleanrl/ppo_continuous_action.py \
    --env-id Hexapod-v0 \
    --total-timesteps 1500000 \
    --num-envs 8 \
    --save-model \
    --seed 564 \
    --learning-rate 1e-3 \
    --exp-name lr1e-3 \
    # --capture-video
