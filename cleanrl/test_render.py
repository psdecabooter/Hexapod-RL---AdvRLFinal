"""
Quick test script to visualize the hexapod environment rendering without training.
Just shows a few steps of random actions to test camera positioning.
"""

import numpy as np
import gymnasium as gym
from hexapod_env import HexapodEnv
import matplotlib.pyplot as plt

def test_render():
    # Create environment with rgb_array rendering
    env = HexapodEnv(render_mode="rgb_array", max_steps=100)
    
    # Reset environment
    obs, info = env.reset()
    
    print("Testing render...")
    print(f"Observation shape: {obs.shape}")
    print(f"Goal position: {env.goal_position}")
    print(f"Robot starting position: {obs[:3]}")
    
    # Create figure for displaying frames
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Take a few steps and capture frames at different points
    steps_to_capture = [0, 10, 20, 30, 40, 50]
    
    for idx, target_step in enumerate(steps_to_capture):
        # Step until we reach target step
        while env.current_step < target_step:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
                break
        
        # Render and display
        frame = env.render()
        axes[idx].imshow(frame)
        axes[idx].set_title(f"Step {env.current_step}\nDistance to goal: {info.get('distance_to_goal', 0):.2f}m")
        axes[idx].axis('off')
        
        print(f"Step {env.current_step}: Distance to goal = {info.get('distance_to_goal', 0):.3f}m")
    
    plt.tight_layout()
    plt.savefig('render_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved render test to 'render_test.png'")
    print("Opening figure...")
    plt.show()
    
    env.close()

if __name__ == "__main__":
    test_render()
