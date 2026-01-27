"""
Test and visualize a trained PPO policy for the hexapod environment.

Usage:
    python test_trained_policy.py runs/Hexapod-v0__ppo_continuous_action__1__1733123456/ppo_continuous_action.cleanrl_model
"""

import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import hexapod_env  # Import to register the environment


# Copy the Agent class from PPO (must match training architecture)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def test_policy(model_path, num_episodes=5, render=True):
    """Test a trained policy."""
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy vectorized env just for Agent initialization
    dummy_env = gym.vector.SyncVectorEnv([lambda: gym.make("Hexapod-v0")])
    agent = Agent(dummy_env).to(device)
    dummy_env.close()
    
    # Load model weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    # Create environment with rendering
    if render:
        eval_env = gym.make("Hexapod-v0", render_mode="human")
    else:
        eval_env = gym.make("Hexapod-v0")
    
    # Load model weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Testing for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get action from policy (deterministic - use mean)
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                action = agent.actor_mean(obs_tensor)
                action = action.cpu().numpy()[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}, "
              f"Final X = {info['x_position']:.3f}m")
    
    print("\n" + "="*60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Distance: {np.mean([info['x_position'] for _ in range(num_episodes)]):.3f}m")
    print("="*60)
    
    eval_env.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_trained_policy.py <path_to_model>")
        print("\nExample:")
        print("  python test_trained_policy.py runs/Hexapod-v0__ppo_continuous_action__1__1733123456/ppo_continuous_action.cleanrl_model")
        print("\nTo find your model, look in the runs/ directory for the latest training run.")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_policy(model_path, num_episodes=5, render=True)
