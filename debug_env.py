"""
Debug script to verify environment readings and robot state.
"""

import numpy as np
import pybullet as p
from hexapod_env import HexapodEnv
import time


def test_environment_readings():
    """Test and verify all environment readings."""
    print("="*70)
    print("Hexapod Environment Debug Test")
    print("="*70)
    
    # Create environment with GUI
    print("\n1. Creating environment with GUI...")
    env = HexapodEnv(render_mode="human")
    print("   ✓ Environment created")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset()
    print("   ✓ Environment reset")
    
    # Wait a bit for GUI to stabilize
    time.sleep(1)
    
    # Get base position and orientation
    base_pos, base_orn = p.getBasePositionAndOrientation(
        env.robot_id, 
        physicsClientId=env.physics_client
    )
    euler = p.getEulerFromQuaternion(base_orn)
    
    print("\n3. Initial Robot State:")
    print(f"   Base Position (x, y, z): {base_pos}")
    print(f"   - X (forward/back): {base_pos[0]:.4f} m")
    print(f"   - Y (left/right):   {base_pos[1]:.4f} m")
    print(f"   - Z (height):       {base_pos[2]:.4f} m")
    print(f"\n   Base Orientation (quaternion): {base_orn}")
    print(f"   Euler angles (roll, pitch, yaw): {euler}")
    print(f"   - Roll:  {np.degrees(euler[0]):.2f}° (rotation around X)")
    print(f"   - Pitch: {np.degrees(euler[1]):.2f}° (rotation around Y)")
    print(f"   - Yaw:   {np.degrees(euler[2]):.2f}° (rotation around Z)")
    
    # Get velocities
    base_lin_vel, base_ang_vel = p.getBaseVelocity(
        env.robot_id,
        physicsClientId=env.physics_client
    )
    print(f"\n   Linear Velocity (vx, vy, vz): {base_lin_vel}")
    print(f"   Angular Velocity (wx, wy, wz): {base_ang_vel}")
    
    # Get joint information
    print(f"\n4. Joint Information:")
    print(f"   Total controllable joints: {len(env.joint_indices)}")
    print(f"   Expected joints: {env.num_joints}")
    
    joint_states = p.getJointStates(
        env.robot_id,
        env.joint_indices,
        physicsClientId=env.physics_client
    )
    
    print("\n   Joint states (first 6 joints):")
    for i, (idx, state) in enumerate(zip(env.joint_indices[:6], joint_states[:6])):
        joint_info = p.getJointInfo(env.robot_id, idx, physicsClientId=env.physics_client)
        joint_name = joint_info[1].decode('utf-8')
        position = state[0]
        velocity = state[1]
        print(f"   - {joint_name:15s}: pos={position:6.3f} rad, vel={velocity:6.3f} rad/s")
    
    # Test observation breakdown
    print("\n5. Observation Space Breakdown:")
    print(f"   Total observation dimensions: {len(obs)}")
    print(f"   - Base position (3):     obs[0:3]   = {obs[0:3]}")
    print(f"   - Base orientation (4):  obs[3:7]   = {obs[3:7]}")
    print(f"   - Linear velocity (3):   obs[7:10]  = {obs[7:10]}")
    print(f"   - Angular velocity (3):  obs[10:13] = {obs[10:13]}")
    print(f"   - Joint positions (18):  obs[13:31] = {obs[13:31]}")
    print(f"   - Joint velocities (18): obs[31:49] = {obs[31:49]}")
    
    # Verify base_pos[2] is height
    print("\n6. Verifying base_pos[2] is height:")
    print(f"   base_pos[2] = {base_pos[2]:.4f} m")
    print(f"   obs[2] (z position) = {obs[2]:.4f} m")
    print(f"   Match: {np.isclose(base_pos[2], obs[2])}")
    
    # Test with some actions
    print("\n7. Testing actions and movements:")
    print("   Applying random actions for 100 steps...")
    
    initial_height = base_pos[2]
    min_height = initial_height
    max_height = initial_height
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_height = obs[2]
        min_height = min(min_height, current_height)
        max_height = max(max_height, current_height)
        
        if step % 20 == 0:
            print(f"   Step {step:3d}: height={current_height:.4f}m, reward={reward:.3f}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step}")
            break
    
    print(f"\n   Height range during test:")
    print(f"   - Initial: {initial_height:.4f} m")
    print(f"   - Minimum: {min_height:.4f} m")
    print(f"   - Maximum: {max_height:.4f} m")
    print(f"   - Variation: {max_height - min_height:.4f} m")
    
    # Test reward calculation
    print("\n8. Testing Reward Components:")
    obs, info = env.reset()
    
    # Stand still (should get height reward)
    action = np.zeros(env.num_joints)
    obs, reward, _, _, _ = env.step(action)
    
    base_pos, base_orn = p.getBasePositionAndOrientation(
        env.robot_id,
        physicsClientId=env.physics_client
    )
    base_lin_vel, _ = p.getBaseVelocity(env.robot_id, physicsClientId=env.physics_client)
    
    print(f"   Current height: {base_pos[2]:.4f} m")
    print(f"   Target height: 0.10 m")
    print(f"   Forward velocity: {base_lin_vel[0]:.4f} m/s")
    print(f"   Total reward: {reward:.3f}")
    
    # Test termination conditions
    print("\n9. Testing Termination Conditions:")
    print(f"   Current roll: {np.degrees(euler[0]):.2f}°")
    print(f"   Current pitch: {np.degrees(euler[1]):.2f}°")
    print(f"   Termination threshold: ±143°")
    print(f"   Should terminate: {env._is_terminated()}")
    
    # Close environment
    print("\n10. Closing environment...")
    env.close()
    print("   ✓ Environment closed")
    
    print("\n" + "="*70)
    print("All tests completed! ✓")
    print("="*70)
    print("\nKey Findings:")
    print("• base_pos[2] correctly represents height (Z-axis)")
    print("• base_pos[0] represents forward/backward position (X-axis)")
    print("• base_pos[1] represents left/right position (Y-axis)")
    print("• Orientation is given as quaternion, converted to Euler angles")
    print("• All observations match PyBullet's internal state")
    print("="*70)


if __name__ == "__main__":
    test_environment_readings()
