"""
PyBullet-based Gymnasium environment for hexapod robot locomotion.
Compatible with all CleanRL continuous action algorithms.

Usage with CleanRL:
    cd cleanrl
    python cleanrl/ppo_continuous_action.py --env-id Hexapod-v0 --total-timesteps 2000000
    python cleanrl/sac_continuous_action.py --env-id Hexapod-v0
    python cleanrl/td3_continuous_action.py --env-id Hexapod-v0
    python cleanrl/ddpg_continuous_action.py --env-id Hexapod-v0

The environment is automatically registered as 'Hexapod-v0' when imported.
"""

import os
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class HexapodEnv(gym.Env):
    """Custom Gymnasium environment for hexapod robot using PyBullet physics."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None, max_steps=1000, control_frequency=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # PyBullet simulation timestep (standard)
        self.simulation_dt = 1.0 / 240.0
        
        # Control frequency parameters (mimics real servo limitations)
        # Set to None for fastest training, or a number (e.g., 30) for realistic servo constraints
        if control_frequency is None:
            # No control frequency limit - fastest training
            self.steps_per_control = 1
        else:
            self.control_frequency = control_frequency  # Hz (commands per second)
            self.control_dt = 1.0 / control_frequency  # time between control updates
            self.steps_per_control = max(1, int(self.control_dt / self.simulation_dt))  # sim steps per control update
        
        # Connect to PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.simulation_dt)
        
        # Load URDF path (relative to this file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, "test_description", "urdf", "test.urdf")
        
        # Joint names for the hexapod (6 legs, 3 joints each = 18 controllable joints)
        self.joint_names = []
        for leg in range(1, 7):
            self.joint_names.extend([
                f"leg{leg}_hip",
                f"leg{leg}_knee", 
                f"leg{leg}_ankle"
            ])
        
        self.num_joints = len(self.joint_names)  # 18 joints
        
        # Action space: continuous control for all 18 joints
        # Each joint can be commanded from -1 to 1 (will be scaled to joint limits)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_joints,), 
            dtype=np.float32
        )
        
        # Observation space: 
        # - base position (3) 
        # - base orientation quaternion (4)
        # - base linear velocity (3)
        # - base angular velocity (3)
        # - joint positions (18)
        # - joint velocities (18)
        # - goal vector (2: X, Y to goal)
        # - distance to goal (1)
        # Total: 52 dimensions
        obs_dim = 3 + 4 + 3 + 3 + self.num_joints + self.num_joints + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.reward_components = {}
        
        # Store previous position for reward calculation
        self.prev_base_pos = None
        
        # Goal position and visualization
        self.goal_position = None
        self.goal_threshold = 0.5  # Distance to consider "reached goal"
        self.goal_visual_id = None
        self.goal_distance = 1.5  # Fixed distance for goal
        
        # Camera tracking
        self.camera_target = None
        
        # Robot and plane IDs
        self.robot_id = None
        self.plane_id = None
        
        # Joint information
        self.joint_indices = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Explicitly remove goal marker if it exists
        if self.goal_visual_id is not None:
            try:
                p.removeBody(self.goal_visual_id, physicsClientId=self.physics_client)
            except:
                pass
        self.goal_visual_id = None
        
        # Reset PyBullet simulation
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        # Load hexapod robot
        # Fix the URDF path issue by replacing package:// with absolute path
        start_pos = [0, 0, 0.15]  # Start slightly above ground
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        try:
            self.robot_id = p.loadURDF(
                self.urdf_path,
                start_pos,
                start_orientation,
                useFixedBase=False,
                physicsClientId=self.physics_client
            )
        except Exception as e:
            print(f"Error loading URDF: {e}")
            print(f"Trying to load from: {self.urdf_path}")
            raise
        
        # Get controllable joint indices
        self.joint_indices = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Only add revolute joints that match our naming scheme
            if joint_type == p.JOINT_REVOLUTE and joint_name in self.joint_names:
                self.joint_indices.append(i)
                self.joint_limits_lower.append(joint_info[8])  # lower limit
                self.joint_limits_upper.append(joint_info[9])  # upper limit
        
        # Sort by name to ensure consistent ordering
        joint_data = sorted(zip(self.joint_indices, self.joint_limits_lower, self.joint_limits_upper,
                               [p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)[1].decode('utf-8') for i in self.joint_indices]),
                           key=lambda x: self.joint_names.index(x[3]))
        
        self.joint_indices = [x[0] for x in joint_data]
        self.joint_limits_lower = np.array([x[1] for x in joint_data])
        self.joint_limits_upper = np.array([x[2] for x in joint_data])
        
        # Verify we found all expected joints
        if len(self.joint_indices) != self.num_joints:
            found_joints = [x[3] for x in joint_data]
            print(f"Warning: Expected {self.num_joints} joints but found {len(self.joint_indices)}")
            print(f"Found joints: {found_joints}")
            # Update num_joints to match what we actually found
            self.num_joints = len(self.joint_indices)
        
        # Initialize joints to a neutral standing position
        for idx in self.joint_indices:
            p.resetJointState(self.robot_id, idx, 0.0, physicsClientId=self.physics_client)
        
        # Enable joint motors with position control
        for idx in self.joint_indices:
            p.setJointMotorControl2(
                self.robot_id,
                idx,
                p.POSITION_CONTROL,
                targetPosition=0,
                force=5.0,
                physicsClientId=self.physics_client
            )
        
        # Let the robot settle
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # Reset state variables
        self.current_step = 0
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        self.prev_base_pos = np.array(base_pos)
        
        # Set goal position: 1.5 meters forward
        self.goal_position = np.array([
            base_pos[0] + self.goal_distance,
            base_pos[1]
        ])
        
        # Create visual marker for goal (flat circle on ground)
        goal_height = 0.01
        goal_radius = self.goal_threshold
        
        goal_vis_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=goal_radius,
            length=goal_height,
            rgbaColor=[0, 1, 0, 0.8],
            physicsClientId=self.physics_client
        )
        
        self.goal_visual_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=goal_vis_shape,
            basePosition=[self.goal_position[0], self.goal_position[1], goal_height/2],
            physicsClientId=self.physics_client
        )
        
        # Initialize camera target at midpoint between robot and goal
        mid_x = (base_pos[0] + self.goal_position[0]) / 2
        mid_y = (base_pos[1] + self.goal_position[1]) / 2
        self.camera_target = np.array([mid_x, mid_y, 0.0])
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        # Ensure action is the right size
        if len(action) != len(self.joint_indices):
            raise ValueError(f"Action size {len(action)} doesn't match number of controllable joints {len(self.joint_indices)}")
        
        # Scale actions from [-1, 1] to actual joint limits
        action = np.clip(action, -1.0, 1.0)
        scaled_action = self.joint_limits_lower + (action + 1.0) / 2.0 * (
            self.joint_limits_upper - self.joint_limits_lower
        )
        
        # Apply actions to joints (only once per control update)
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=scaled_action[i],
                force=5.0,
                physicsClientId=self.physics_client
            )
        
        # Step simulation for the duration of one control period
        # This enforces the control frequency limit (e.g., 20 Hz means ~12 sim steps between commands)
        for _ in range(self.steps_per_control):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        self.current_step += 1
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation from the environment."""
        # Base pose and velocity
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        
        # Joint states
        joint_states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.physics_client)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # Goal-related observations
        goal_vector = self.goal_position - np.array([base_pos[0], base_pos[1]])
        distance_to_goal = np.linalg.norm(goal_vector)
        
        # Concatenate all observations
        observation = np.concatenate([
            np.array(base_pos),           # 3
            np.array(base_orn),           # 4
            np.array(base_lin_vel),       # 3
            np.array(base_ang_vel),       # 3
            np.array(joint_positions),    # 18
            np.array(joint_velocities),   # 18
            goal_vector,                  # 2
            [distance_to_goal]            # 1
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward(self):
        """Calculate reward based on robot's behavior."""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        
        # Calculate displacement vector
        displacement = np.array([
            base_pos[0] - self.prev_base_pos[0],  # X (forward/backward)
            base_pos[1] - self.prev_base_pos[1],  # Y (left/right)
        ])
        
        # Calculate total distance moved (in 2D horizontal plane)
        distance_moved = np.linalg.norm(displacement)
        
        # Goal direction is forward along X-axis
        goal_direction = np.array([1.0, 0.0])
        
        # Calculate directional alignment for displacement
        if distance_moved > 1e-6:  # Avoid division by zero
            movement_direction = displacement / distance_moved
            # Dot product gives cos(angle) between movement and goal
            directional_alignment = np.dot(movement_direction, goal_direction)
        else:
            # No movement
            directional_alignment = 0.0
        
        # Primary reward: distance scaled by directional alignment
        # - Moving forward (0°): alignment = 1.0 → full reward
        # - Moving sideways (90°): alignment = 0.0 → zero reward  
        # - Moving backward (180°): alignment = -1.0 → negative reward
        # Subtract constant to create pressure: must move >0.001m forward per step to break even
        distance_reward = distance_moved * directional_alignment * 100.0 - 0.5
        
        # Velocity reward: encourages moving fast in the goal direction
        velocity_2d = np.array([base_lin_vel[0], base_lin_vel[1]])
        speed = np.linalg.norm(velocity_2d)
        
        if speed > 1e-6:
            velocity_direction = velocity_2d / speed
            velocity_alignment = np.dot(velocity_direction, goal_direction)
        else:
            velocity_alignment = 0.0
        
        # Reward fast forward movement
        velocity_reward = speed * velocity_alignment * 5.0
        
        # Stability penalty (staying upright)
        euler = p.getEulerFromQuaternion(base_orn)
        roll, pitch, yaw = euler
        stability_penalty = -1.0 * (abs(roll) + abs(pitch))
        
        # Energy efficiency penalty (smooth, efficient movements)
        joint_states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.physics_client)
        joint_velocities = np.array([state[1] for state in joint_states])
        energy_penalty = -0.0005 * np.sum(np.square(joint_velocities))
        
        # Total reward
        reward = distance_reward + velocity_reward + stability_penalty + energy_penalty
        
        # Update previous position
        self.prev_base_pos = np.array(base_pos)
        
        self.reward_components = {
            "reward/distance": distance_reward,
            "reward/velocity": velocity_reward,
            "reward/stability_penalty": stability_penalty,
            "reward/energy_penalty": energy_penalty,
            "reward/total": reward,
        }
        
        return reward
    
    def _is_terminated(self):
        """Check if episode should terminate."""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        
        # Only terminate if robot flips completely over
        # Don't terminate on low height - let the robot learn to avoid it naturally
        euler = p.getEulerFromQuaternion(base_orn)
        roll, pitch, yaw = euler
        if abs(roll) > 1.57 or abs(pitch) > 1.57:  # ~90 degrees
            return True
        
        return False
    
    def _get_info(self):
        """Get additional info for logging."""
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        base_lin_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        
        # Calculate distance to goal
        to_goal = self.goal_position - np.array([base_pos[0], base_pos[1]])
        distance_to_goal = np.linalg.norm(to_goal)
        
        return {
            "x_position": base_pos[0],
            "y_position": base_pos[1],
            "z_position": base_pos[2],
            "x_velocity": base_lin_vel[0],
            "distance_to_goal": distance_to_goal,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Get robot position
            base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
            robot_2d = np.array([base_pos[0], base_pos[1]])
            
            # Calculate midpoint between robot and goal for camera framing
            mid_x = (base_pos[0] + self.goal_position[0]) / 2
            mid_y = (base_pos[1] + self.goal_position[1]) / 2
            
            # Only move camera if robot exits the dead zone from the midpoint
            camera_to_robot = robot_2d - np.array([mid_x, mid_y])
            distance_from_camera_center = np.linalg.norm(camera_to_robot)
            
            if distance_from_camera_center > self.camera_deadzone_radius:
                # Robot has moved too far, update camera target to keep both robot and goal visible
                # Recalculate midpoint as camera target
                self.camera_target[:2] = np.array([mid_x, mid_y])
            else:
                # Keep current camera target for stability
                pass
            
            # Calculate distance between robot and goal to determine camera zoom
            to_goal = self.goal_position - np.array([base_pos[0], base_pos[1]])
            distance_to_goal = np.linalg.norm(to_goal)
            
            # Camera distance based on how far apart robot and goal are
            # Need to see both, so distance scales with their separation
            # Closer camera for better detail visibility
            camera_distance = max(1.5, distance_to_goal * 0.75)
            
            # Camera points at the target position (midpoint between robot and goal)
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[self.camera_target[0], self.camera_target[1], base_pos[2]],
                distance=camera_distance,
                yaw=-90,  # Side view from the other side - robot moves right to left on screen
                pitch=-20,  # Look down at 20 degrees
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.physics_client
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=70, aspect=16/9, nearVal=0.1, farVal=100.0  # Wider FOV, 16:9 aspect
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=720,  # Lower resolution for faster startup/rendering
                height=640,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER,  # Software renderer - more stable in WSL
                physicsClientId=self.physics_client
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (640, 720, 4))
            rgb_array = rgb_array[:, :, :3]
            return rgb_array
        elif self.render_mode == "human":
            # GUI mode handles rendering automatically
            pass
    
    def close(self):
        """Clean up the environment."""
        if self.physics_client >= 0:
            # Remove goal marker if it exists
            if self.goal_visual_id is not None:
                try:
                    p.removeBody(self.goal_visual_id, physicsClientId=self.physics_client)
                except:
                    pass
            # Disconnect from PyBullet
            try:
                p.disconnect(self.physics_client)
            except:
                pass
            self.physics_client = -1
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


# Register the environment with Gymnasium
gym.register(
    id='Hexapod-v0',
    entry_point='hexapod_env:HexapodEnv',
    max_episode_steps=1000,
)
