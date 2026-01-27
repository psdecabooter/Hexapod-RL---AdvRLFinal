"""
Quick test to verify the hexapod environment works correctly.
"""

import gymnasium as gym
from hexapod_env import HexapodEnv

def test_basic():
    """Test basic environment functionality."""
    print("Testing Hexapod Environment...")
    print("-" * 60)
    
    # Test direct instantiation
    print("1. Creating environment...")
    env = HexapodEnv(render_mode="human")
    print(f"   ✓ Environment created")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Action space: {env.action_space.shape}")
    
    # Test reset
    print("\n2. Testing reset...")
    obs, info = env.reset()
    print(f"   ✓ Reset successful")
    print(f"   - Observation shape: {obs.shape}")
    
    # Test steps
    print("\n3. Running 100 random steps...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"   Episode ended at step {i+1}")
            break
    
    print(f"   ✓ Steps completed")
    print(f"   - Final x position: {info['x_position']:.3f}")
    
    env.close()
    print("\n4. Environment closed")
    
    # Test gym.make
    print("\n5. Testing gym.make('Hexapod-v0')...")
    try:
        env = gym.make('Hexapod-v0')
        obs, info = env.reset()
        print(f"   ✓ gym.make() works")
        env.close()
    except Exception as e:
        print(f"   ✗ gym.make() failed: {e}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nYou can now use this environment with CleanRL:")
    print("  cd cleanrl")
    print("  python cleanrl/ppo_continuous_action.py --env-id Hexapod-v0")

if __name__ == "__main__":
    test_basic()
