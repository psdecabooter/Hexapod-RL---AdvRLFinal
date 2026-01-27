"""
Verify actual link lengths by loading URDF in PyBullet and measuring joint positions.
This shows what PyBullet actually sees from the URDF.
"""

import pybullet as p
import pybullet_data
import numpy as np
import os

# Connect to PyBullet
physics_client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load URDF
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "test_description", "urdf", "test.urdf")

robot_id = p.loadURDF(urdf_path, [0, 0, 0.5], useFixedBase=False)

print("="*70)
print("URDF Link Length Verification")
print("="*70)

# Get all joint info
num_joints = p.getNumJoints(robot_id)
print(f"\nTotal joints in URDF: {num_joints}")

# Find leg1 joints
leg1_joints = {}
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    if 'leg1' in joint_name and joint_name in ['leg1_hip', 'leg1_knee', 'leg1_ankle']:
        leg1_joints[joint_name] = {
            'index': i,
            'type': joint_info[2],
            'parent_link': joint_info[16],
        }
        print(f"\nFound: {joint_name} (index {i})")

# Reset all joints to zero position
for i in range(num_joints):
    p.resetJointState(robot_id, i, 0.0)

# Let physics settle
for _ in range(100):
    p.stepSimulation()

print("\n" + "="*70)
print("Measuring 3D positions with joints at zero position:")
print("="*70)

# Get link states for leg1
if 'leg1_hip' in leg1_joints:
    hip_idx = leg1_joints['leg1_hip']['index']
    hip_link_state = p.getLinkState(robot_id, hip_idx)
    hip_pos = np.array(hip_link_state[0])
    print(f"\nHip joint (leg1_hip) position: {hip_pos}")
    print(f"  X: {hip_pos[0]*1000:.2f} mm")
    print(f"  Y: {hip_pos[1]*1000:.2f} mm")
    print(f"  Z: {hip_pos[2]*1000:.2f} mm")

if 'leg1_knee' in leg1_joints:
    knee_idx = leg1_joints['leg1_knee']['index']
    knee_link_state = p.getLinkState(robot_id, knee_idx)
    knee_pos = np.array(knee_link_state[0])
    print(f"\nKnee joint (leg1_knee) position: {knee_pos}")
    print(f"  X: {knee_pos[0]*1000:.2f} mm")
    print(f"  Y: {knee_pos[1]*1000:.2f} mm")
    print(f"  Z: {knee_pos[2]*1000:.2f} mm")
    
    if 'leg1_hip' in leg1_joints:
        hip_to_knee = np.linalg.norm(knee_pos - hip_pos)
        print(f"\n>>> Hip to Knee distance: {hip_to_knee*1000:.2f} mm")

if 'leg1_ankle' in leg1_joints:
    ankle_idx = leg1_joints['leg1_ankle']['index']
    ankle_link_state = p.getLinkState(robot_id, ankle_idx)
    ankle_pos = np.array(ankle_link_state[0])
    print(f"\nAnkle joint (leg1_ankle) position: {ankle_pos}")
    print(f"  X: {ankle_pos[0]*1000:.2f} mm")
    print(f"  Y: {ankle_pos[1]*1000:.2f} mm")
    print(f"  Z: {ankle_pos[2]*1000:.2f} mm")
    
    if 'leg1_knee' in leg1_joints:
        knee_to_ankle = np.linalg.norm(ankle_pos - knee_pos)
        print(f"\n>>> Knee to Ankle distance: {knee_to_ankle*1000:.2f} mm")

# Find the leg_link_1 (tibia/tip)
print("\n" + "="*70)
print("Finding leg tip:")
print("="*70)

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    link_name = joint_info[12].decode('utf-8')
    if 'leg_link_1' in link_name:
        # This is the tibia link - get its end position
        link_state = p.getLinkState(robot_id, i)
        tip_pos = np.array(link_state[0])
        print(f"\nLeg link (leg_link_1) center: {tip_pos}")
        print(f"  X: {tip_pos[0]*1000:.2f} mm")
        print(f"  Y: {tip_pos[1]*1000:.2f} mm")
        print(f"  Z: {tip_pos[2]*1000:.2f} mm")
        
        if 'leg1_ankle' in leg1_joints:
            ankle_to_tip_center = np.linalg.norm(tip_pos - ankle_pos)
            print(f"\n>>> Ankle to leg link CENTER: {ankle_to_tip_center*1000:.2f} mm")
            print(f"    (Actual tip is likely further - this is just the link's COM)")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print("These are the ACTUAL 3D distances PyBullet sees from your URDF.")
print("Compare these with your Fusion 360 measurements.")
print("="*70)

p.disconnect()
