#!/bin/bash
# Quick training script for hexapod with model saving enabled
# NOTE: num-envs MUST be 1 for PyBullet environments to prevent memory leaks and crashes

python cleanrl/ppo_continuous_action.py \
    --env-id Hexapod-v0 \
    --total-timesteps 2500000 \
    --num-envs 8 \
    --save-model \
    # --capture-video
