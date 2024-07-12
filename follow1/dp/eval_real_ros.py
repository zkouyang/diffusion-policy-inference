"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

import sys

sys.path.append("/home/dc/mambaforge/envs/robodiff/lib/python3.9/site-packages")
# sys.path.append(
#     "/media/dc/CLEAR/arx/dp_inference/dp"
# )

import time
import math
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st


from common.precise_sleep import precise_wait
from utils.real_inference_utils import get_real_obs_resolution, get_real_obs_dict
from codebase.diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply
from codebase.diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from codebase.diffusion_policy.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from utils.cv2_utils import get_image_transform

from codebase.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from arm_control.msg import JointInformation
from arm_control.msg import JointControl
from arm_control.msg import PosCmd
from sensor_msgs.msg import Image
from threading import Lock
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

gripper_width_min = -0.016
gripper_width_max = 0.4
actual_gripper_width_min = 0.0
actual_gripper_width_max = 5.0

OmegaConf.register_new_resolver("eval", eval, replace=True)
np.set_printoptions(suppress=True)

@click.command()
@click.option(
    "--input_path",
    "-ip",
    required=True,
    help="Path to checkpoint",
)
@click.option(
    "--frequency",
    "-f",
    default=10,
    type=int,
    help="control frequency",
)


@click.option(
    "--steps_per_inference",
    "-si",
    default=8,
    type=int,
    help="Action horizon for inference.",
)


# @profile
def main(
    input_path,
    frequency,
    steps_per_inference,
):
    global obs_ring_buffer

    dt = 1 / frequency
    video_capture_fps = 30
    max_obs_buffer_size = 30

    

    # load checkpoint
    ckpt_path = input_path
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    print(cfg)
    # exit()
    cfg._target_ = "codebase.diffusion_policy." + cfg._target_  # can delete

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda:0")
    policy.eval().to(device)

    ## set inference params
    policy.num_inference_steps = 16  # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    
    # shared memory
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    examples = dict()
    examples["mid"] = np.empty(shape=obs_res[::-1] + (3,), dtype=np.uint8)
    examples["right"] = np.empty(shape=obs_res[::-1] + (3,), dtype=np.uint8)
    examples["eef_qpos"] = np.empty(shape=(7,), dtype=np.float64)
    examples["qpos"] = np.empty(shape=(7,), dtype=np.float64)
    examples["timestamp"] = 0.0
    obs_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        shm_manager=shm_manager,
        examples=examples,
        get_max_k=max_obs_buffer_size,
        get_time_budget=0.2,
        put_desired_frequency=video_capture_fps,
    )

    # ros init and subscriber
    rospy.init_node("eval_real_ros")
    eef_qpos = Subscriber("follow2_pos_back", PosCmd)
    qpos = Subscriber("joint_information2", JointInformation)
    mid = Subscriber("mid_camera", Image)
    right = Subscriber("right_camera", Image)
    # control_robot2 = rospy.Publisher("test_right", JointControl, queue_size=10)
    control_robot2 = rospy.Publisher("follow_joint_control_2", JointControl, queue_size=10)
    ats = ApproximateTimeSynchronizer(
        [eef_qpos, qpos, mid, right], queue_size=10, slop=0.1
    )
    ats.registerCallback(callback)
    rate = rospy.Rate(frequency)


    # data ??
    last_data = None
    right_control = JointControl()
    

    # start
    # ========== policy control loop ==============
    # start episode
    policy.reset()
    start_delay = 1.0
    eval_t_start = time.time() + start_delay
    t_start = time.monotonic() + start_delay
    frame_latency = 1/30
    precise_wait(eval_t_start - frame_latency, time_func=time.time)
    print("Started!")
    iter_idx = 0

    # inference loop
    while not rospy.is_shutdown():
        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

        # get observation
        k = math.ceil(n_obs_steps * (video_capture_fps / frequency))
        last_data = obs_ring_buffer.get_last_k(k=k, out=last_data)
        last_timestamp = last_data["timestamp"][-1]
        obs_align_timestamps = last_timestamp - (np.arange(n_obs_steps)[::-1] * dt)

        obs_dict = dict()
        this_timestamps = last_data["timestamp"]
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps <= t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)
        for key in last_data.keys():
            obs_dict[key] = last_data[key][this_idxs]

        obs_timestamps = obs_dict["timestamp"]
        print("Got Observation!")

        # run inference
        with torch.no_grad():
            test_t_start = time.perf_counter()
            obs_dict_np = get_real_obs_dict(
                env_obs=obs_dict, shape_meta=cfg.task.shape_meta)
            obs_dict = dict_apply(obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(torch.device("cuda:0")))

            # image = obs_dict['right'][0, 1, :, :, :]
            # image_converted = np.transpose(image.cpu().numpy(), (1, 2, 0))
            # plt.imshow(image_converted)
            # plt.show()
            # import ipdb;ipdb.set_trace()
            # exit()
            print(obs_dict['qpos'])
            result = policy.predict_action(obs_dict)
            
            action = result["action"][0].detach().to("cpu").numpy()
            # print(action)
            # print("----------------------------")
            print(f"Inference latency {steps_per_inference/(time.perf_counter() - test_t_start)}")

        # preprocess action
        action = action[:steps_per_inference, :]
        action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]

        action_exec_latency = 0.01
        curr_time = time.time()
        is_new = action_timestamps > (curr_time + action_exec_latency)

        if np.sum(is_new) == 0:
            action = action[[-1]]
            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
            action_timestamp = eval_t_start + (next_step_idx) * dt
            print("Over budget", action_timestamp - curr_time)
            action_timestamps = np.array([action_timestamp])
        else:
            action = action[is_new]
            action_timestamps = action_timestamps[is_new]
            
        # execute actions
        print("Execute Action!")
        print(action)
        print("==================================")
        for item in action:
            right_control.joint_pos = item
            
            
            # control_robot2.publish(right_control)
            rate.sleep()

        precise_wait(t_cycle_end - frame_latency)
        iter_idx += steps_per_inference

        


def callback(eef_qpos, qpos, image_mid, image_right):
    global obs_ring_buffer

    mid = image_mid
    right = image_right
    bridge = CvBridge()
    receive_time = time.time()

    obs_data = dict()
    obs_data["eef_qpos"] = np.array(
        [
            eef_qpos.x,
            eef_qpos.y,
            eef_qpos.z,
            eef_qpos.roll,
            eef_qpos.pitch,
            eef_qpos.yaw,
            eef_qpos.gripper,
        ]
    )
    gripper_width = qpos.joint_pos[6]
    gripper_width = (gripper_width - actual_gripper_width_min) / (actual_gripper_width_max - actual_gripper_width_min) * \
                    (gripper_width_max - gripper_width_min) + gripper_width_min
    obs_data["qpos"] = np.array(
        [
            qpos.joint_pos[0],
            qpos.joint_pos[1],
            qpos.joint_pos[2],
            qpos.joint_pos[3],
            qpos.joint_pos[4],
            qpos.joint_pos[5],
            gripper_width
        ]
    )

    # process images observation
    mid = bridge.imgmsg_to_cv2(mid, "bgr8")
    right = bridge.imgmsg_to_cv2(right, "bgr8")
    obs_data["mid"] = transform(mid)
    obs_data["right"] = transform(right)
    obs_data["timestamp"] = receive_time

    # show pictures
    canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
    canvas[:, :640, :] = mid
    canvas[:, 640:1280, :] = right

    cv2.imshow('Multi Camera Viewer', canvas)
    if cv2.waitKey(1) == ord("q"):
        rospy.signal_shutdown("User requested shutdown")

    # put data
    put_data = obs_data
    obs_ring_buffer.put(put_data, wait=False)


def transform(data, video_capture_resolution=(640, 480), obs_image_resolution=(224, 224)):
    color_tf = get_image_transform(
                input_res=video_capture_resolution,
                output_res=obs_image_resolution,
                # obs output rgb
                bgr_to_rgb=False,
            )
    
    tf_data = color_tf(data)
    return tf_data

if __name__ == "__main__":
    main()
