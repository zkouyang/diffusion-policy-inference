#!/home/shawn/anaconda3/envs/umi/bin/python
import sys
sys.path.append("/home/shawn/anaconda3/envs/umi/lib/python3.9/site-packages")
sys.path.append("/home/shawn/Desktop/arx-umi/follow_control/follow1/src/arm_control/scripts/universal_manipulation_interface")
if '/usr/lib/python3/dist-packages' in sys.path:
    sys.path.remove('/usr/lib/python3/dist-packages')
import os
import rospy
import time
import yaml
import threading
import signal
import numpy as np
import pickle
from queue import Queue
from typing import Optional, List
from arm_control.msg import PosCmd
from multiprocessing.managers import SharedMemoryManager
import multiprocessing.shared_memory as sm
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.shared_memory.shared_memory_util import clear_shared_memory
from ros_real_world.ros_pose_util import (
    pose_to_mat, mat_to_pose, rotvec_to_matrix)

T_ur5_arx5 = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

def spin():
    rospy.spin()

def signal_handler(sig, frame):
    print('Ctrl+C pressed, exiting.')
    rospy.signal_shutdown('Ctrl+C pressed')

def has_data(shm):
    return any(byte != 0 for byte in shm.buf)          

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = rotvec_to_matrix(ee_pose[3:6])
    transformed_keypoints = (
        np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    )
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

class RobotRosController:
    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_name=None,
            tcp_offset = None,
            frequency=200,
            receive_latency=0.0,
            get_max_k=None,
            init_joints=False,):
        assert robot_name in ['robot0', 'robot1']
        self.publisher = rospy.Publisher(f"follow_pos_cmd_{int(robot_name[-1]) + 1}", PosCmd, queue_size=10)
        self.subscriber = rospy.Subscriber(f"follow{int(robot_name[-1]) + 1}_pos_back", PosCmd, self.callback)
        self.obs_shm = sm.SharedMemory(name="robot_obs", create=True, size = 40960)
        self.shm_manager=shm_manager    
        self.frequency=frequency
        self.receive_latency=receive_latency
        self.init_joints=init_joints
        self.robot_name=robot_name
        self.tcp_offset=tcp_offset
        
        if init_joints:
            self.init_pose()

        if get_max_k is None:
            get_max_k = int(frequency * 2)
        
        # build ring buffer
        receive_keys= [
            f'{robot_name}_eef_pos',
            f'{robot_name}_eef_rot_axis_angle',
            f'{robot_name}_gripper_width'
        ]
        example=dict()
        example[f'{robot_name}_eef_pos']=np.zeros(3)
        example[f'{robot_name}_eef_rot_axis_angle']=np.zeros(3)
        example[f'{robot_name}_gripper_width']=np.zeros(1)
        example['robot_timestamp'] = time.time()
        example['robot_receive_timestamp'] = time.time()
        
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

        self.gripper_scale = 51.1

    def init_pose(self):
        print(f"==========init {self.robot_name}'s pose==========")
        pos_cmd_msg = PosCmd()
        pos_cmd_msg.x = 0
        pos_cmd_msg.y = 0
        pos_cmd_msg.z = 0
        pos_cmd_msg.roll = 0
        pos_cmd_msg.pitch = 0
        pos_cmd_msg.yaw = 0
        pos_cmd_msg.gripper = 0
        self.publisher.publish(pos_cmd_msg)
        rospy.sleep(5.0)

    def callback(self, robot_pose_msg):
        t_recv = time.time()
        state = dict()
        
        # print("Robot pose received!!!!")
        state['robot_receive_timestamp'] = t_recv
        state['robot_timestamp'] = t_recv - self.receive_latency
        pose_ur5_tcp = mat_to_pose(T_ur5_arx5 @ \
                                   pose_to_mat(np.array([robot_pose_msg.x, robot_pose_msg.y, robot_pose_msg.z, robot_pose_msg.roll, robot_pose_msg.pitch, robot_pose_msg.yaw], dtype=np.float32)) @ \
                                   pose_to_mat(self.tcp_offset) @ \
                                   np.linalg.inv(T_ur5_arx5)
                                   )
        state[f'{self.robot_name}_eef_pos']= pose_ur5_tcp[:3]
        state[f'{self.robot_name}_eef_rot_axis_angle']= pose_ur5_tcp[3:]
        
        
        # # for test
        # print("pose_ur5_tcp: ", pose_ur5_tcp)
        # R_ur5_arx5 = T_ur5_arx5[:3, :3]
        # pos = R_ur5_arx5 @ np.array([robot_pose_msg.x, robot_pose_msg.y, robot_pose_msg.z], dtype=np.float32)
        # rot = R_ur5_arx5 @ np.array([robot_pose_msg.roll, robot_pose_msg.pitch, robot_pose_msg.yaw], dtype=np.float32)
        # print("pos: ", pos, "rot: ", rot)
        
        state[f'{self.robot_name}_gripper_width']=np.array([robot_pose_msg.gripper / self.gripper_scale], dtype=np.float32)
        self.ring_buffer.put(state)
        ##TODO only consider one robot here
        obs = self.ring_buffer.get_all()
        # print(obs)
        # print("obs: ", obs['robot0_eef_pos'][0])
        serialized_obs_data = pickle.dumps(obs)
        self.obs_shm.buf[:len(serialized_obs_data)] = serialized_obs_data

    def exec_action(self):
        ##TODO
        pass
        

if __name__ == '__main__':
    rospy.init_node("robot_ros_controller", anonymous=True)
    rate = rospy.Rate(10)

    # get ros param
    robot_config = rospy.get_param("~robot_config", "/home/shawn/Desktop/arx-umi/follow_control/follow1/src/arm_control/scripts/universal_manipulation_interface/example/ros_robots_config.yaml")
    num_robot = 0

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    robots_config = robot_config_data['robots']

    spin_thread = threading.Thread(target = spin)
    spin_thread.start()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    with SharedMemoryManager() as shm_manager:
        robots: List[RobotRosController] = list()
        for rc in robots_config:
            assert rc['robot_name'] in ['robot0', 'robot1']
            this_robot = RobotRosController(
                    shm_manager=shm_manager,
                    frequency=200, 
                    tcp_offset=np.array([rc['tcp_offset'], 0, 0, 0, 0, 0], dtype=np.float32),
                    receive_latency=rc['robot_obs_latency'],
                    robot_name=rc['robot_name'],
                    init_joints=False,
                )
            num_robot += 1
            robots.append(this_robot)
        last_robots_data = list()
        action_shm = sm.SharedMemory(name="robot_action", create=True, size = 2048)
        action_queue = Queue()

        print("robot controller start!")
        # np.set_printoptions(precision=6)
        while not rospy.is_shutdown():
            if has_data(action_shm):
                for robot_idx, rc in enumerate(robots_config):
                    # receive actions from umi_ros_env
                    serialized_action_data = action_shm.buf[:action_shm.size]
                    predicted_actions = pickle.loads(serialized_action_data)[f'actions_{robot_idx}']
                    predicted_timestamps = pickle.loads(serialized_action_data)[f'timestamps_{robot_idx}']
                    
                    
                    
                    cur_time = time.time()
                    
                    print("action:", predicted_actions)
                    
                    # filter out some actions due to shm's latency
                    actions = predicted_actions[predicted_timestamps - cur_time > 0]
                    timestamps = predicted_timestamps[predicted_timestamps - cur_time > 0]
                    # print(timestamps)
                    assert actions.shape[0] == timestamps.shape[0]

                    if actions.shape[0] > 0:
                        rospy.sleep(timestamps[0] - cur_time)
                        for i in range(16 - (16 - actions.shape[0])):
                            control_msg = PosCmd()
                            pose_ur5_tcp = np.array([actions[i][0], actions[i][1], actions[i][2], actions[i][3], actions[i][4], actions[i][5]], dtype=np.float32) 
                            pose_arx5_eef = mat_to_pose(np.linalg.inv(T_ur5_arx5) @ \
                                                        pose_to_mat(pose_ur5_tcp) @ \
                                                        T_ur5_arx5 @ \
                                                        np.linalg.inv(pose_to_mat(robots[0].tcp_offset)))
                            # print("before: ", pose_arx5_eef[2])
                            solve_table_collision(ee_pose=pose_arx5_eef[robot_idx * 7 : robot_idx * 7 + 6],
                                                  gripper_width=actions[i][robot_idx * 7 + 6],
                                                  height_threshold=robots_config[robot_idx]["height_threshold"])
                            # print("after: ", pose_arx5_eef[2])
                            control_msg.x = pose_arx5_eef[0]
                            control_msg.y = pose_arx5_eef[1]
                            control_msg.z = pose_arx5_eef[2]
                            control_msg.roll = pose_arx5_eef[3]
                            control_msg.pitch = pose_arx5_eef[4]
                            control_msg.yaw = pose_arx5_eef[5]
                            
                            
                            # # for test
                            # print("pose_arx5_eef: ", pose_arx5_eef)
                            # R_ur5_arx5 = T_ur5_arx5[:3, :3]
                            # print(R_ur5_arx5)
                            # pos = np.array([actions[i][0], actions[i][1], actions[i][2]])
                            # rot = np.array([actions[i][3], actions[i][4], actions[i][5]])
                            # pos = np.linalg.inv(R_ur5_arx5) @ pos
                            # rot = np.linalg.inv(R_ur5_arx5) @ rot
                            # print("pos: ", pos, "rot: ", rot)
                            
                            control_msg.gripper = actions[i][6] * robots[0].gripper_scale
                            robots[robot_idx].publisher.publish(control_msg)   

                            rate.sleep()
                clear_shared_memory(action_shm)
            rate.sleep()
        
            





