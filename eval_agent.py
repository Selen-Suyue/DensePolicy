"""
Evaluation Agent.
"""

import time
import numpy as np
from device.robot.flexiv import FlexivRobot
from utils.transformation import xyz_rot_transform
# from device.gripper.dahuan import DahuanModbusGripper
from device.gripper.robotiq import Robotiq2FGripper
from device.camera.realsense import RealSenseRGBDCamera


class Agent:
    """
    Evaluation agent with Flexiv arm, Dahuan gripper and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        robot_ip,
        pc_ip,
        gripper_port,
        camera_serial,
        **kwargs
    ): 
        self.camera_serial = camera_serial

        print("Init robot, gripper, and camera.")
        self.robot = FlexivRobot(robot_ip_address = robot_ip, pc_ip_address = pc_ip)
        self.robot.send_tcp_pose(self.ready_pose)
        time.sleep(1.5)
        
        self.gripper = Robotiq2FGripper(port = gripper_port)
        # self.gripper.set_force(30)
        self.gripper.close_gripper()
        time.sleep(0.5)

        self.camera = RealSenseRGBDCamera(serial = camera_serial)
        for _ in range(30): 
            self.camera.get_rgbd_image()
        print("Initialization Finished.")
    
    @property
    def intrinsics(self):
        return np.array([[915.384521484375, 0, 633.3715209960938, 0],
                        [0, 914.9421997070312, 354.1505432128906, 0],
                        [0, 0, 1, 0]])
    
    @property
    def ready_pose(self):
        return np.array([0.5, 0.0, 0.17, 0.0, 0.0, 1.0, 0.0])

    @property
    def ready_rot_6d(self):
        return np.array([-1, 0, 0, 0, 1, 0])

    def get_observation(self):
        colors, depths = self.camera.get_rgbd_image()
        return colors, depths
    
    def get_tcp_pose(self):
        return np.array(self.robot.get_tcp_pose(), dtype=np.float32)

    def get_gripper_width(self):
        return np.array(self.gripper.get_info()[0:1], dtype=np.float32)
    
    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention = None, blocking = False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )
        self.robot.send_tcp_pose(tcp_pose)
        if blocking:
            time.sleep(0.1)
    
    def set_gripper_width(self, width, blocking = False):
        print(width, end=" ")
        width = int(np.clip((1.0 - width / 0.085) * 255., 0, 255))
        print(width)
        self.gripper.action(width)
        if blocking:
            time.sleep(0.5)
    
    def stop(self):
        self.robot.stop()
    