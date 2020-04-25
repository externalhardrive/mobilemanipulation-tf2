import os
import time
from os.path import expanduser
import sys

import gym
import numpy as np
import pybullet
import pybullet_data
from gym import spaces
from gym.utils import seeding
from sklearn.linear_model import LinearRegression
from . import bullet_client

from .utils import *

IMAGE_SIZE = 100
# GRASP_CLASSIFIER = load_convnet('classifier/model', IMAGE_SIZE, relative_path=True)

class Viewer:
    def __init__(self, p, camera_pos, look_pos, fov=25, near_pos=.02, far_pos=1.):
        self.p = p
        self.proj_matrix = self.p.computeProjectionMatrixFOV(fov, 1, near_pos, far_pos)
        self.update(camera_pos, look_pos)

    def update(self, camera_pos, look_pos):
        self.camera_pos = np.array(camera_pos)
        self.look_pos = np.array(look_pos)
        self.view_matrix = self.p.computeViewMatrix(self.camera_pos, self.look_pos, [0,0,1])

    def get_image(self, width, height):
        _, _, image, _, _ = self.p.getCameraImage(width, height, self.view_matrix, self.proj_matrix)
        return np.reshape(image, (height, width, 4))[:,:,:3]


class PybulletInterface:

    # START_JOINTS = np.array([0., -0.6, 1.3, 0.5, 1.6])
    # # PREGRASP_POS = np.array((0.4568896949291229, -0.00021789505262859166, 0.3259587585926056))
    # PREGRASP_POS = np.array((0.5, 0, 0.3))
    # DOWN_QUAT = np.array((0.00011637622083071619, 0.6645175218582153, 0.00046503773774020374, 0.7472725510597229))
    # BLOCK_POS = np.array([.45, 0., .02])
    # GRIPPER_REF_POS = np.array([.45, 0., .0])
    # # GRIPPER_REF_POS = np.array([10, 0., 0.])
    # CAMERA_LOOK_POS = np.array([0.5, 0., .2])
    # # BLOCK_POS = np.array([0.5, 0, 0.2])

    def __init__(self, **params):
        defaults = {
            "renders": False, 
            "grayscale": False, 
            "step_duration": 1/60, 
            "start_joints": np.array([0., -0.6, 1.3, 0.5, 1.6]),
            "pregrasp_pos": np.array([0.42, 0, 0.3]),
            "down_quat": np.array([0.0, 0.7071067811865475, 0.0, 0.7071067811865476]),
            "block_pos": np.array([.45, 0., .02]),
            "gripper_ref_pos": np.array([.45, 0., .0]),
            "camera_look_pos": np.array([0.5, 0., .2]),
            "image_size": IMAGE_SIZE,
            "camera_fov": 60
        }
        defaults.update(params)
        self.params = defaults

        self.renders = self.params["renders"]
        self.recording = False
        self.grayscale = self.params["grayscale"]
        self.step_duration = self.params["step_duration"]

        if self.renders:
            self.p = bullet_client.BulletClient(
                connection_mode=pybullet.GUI)
            print("LOCOBOT: STARTED RENDER MODE")
        else:
            self.p = bullet_client.BulletClient()

        self.frames = []

        self.max_objects = 1024
        self.num_objects = 0
        self.object_dict = {}
        self.default_ori = self.p.getQuaternionFromEuler([0,0,0])

        self.robot_urdf = URDF["locobot"]
        self.block_urdf = URDF["miniblock"]

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.8)

        self.plane_id = self.p.loadURDF("plane.urdf")
        # self.plane_visual_id = self.p.createVisualShape(self.p.GEOM_PLANE, rgbaColor=[0.9, 0.9, 0.9, 1], visualFramePosition=[0, 0, 0])
        # self.plane_collision_id = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[100, 100, 10], collisionFramePosition=[0, 0, -10])
        # self.plane_id = self.p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
        # self.plane_id = self.p.loadURDF(URDF["plane"], useMaximalCoordinates=True)

        # self.under_plane_id = self.p.loadURDF("plane.urdf")
        # self.p.resetBasePositionAndOrientation(self.under_plane_id, [0, 0, -10], self.default_ori)
        # or set mass to 0

        # Load robot
        self.robot = self.p.loadURDF(self.robot_urdf, useFixedBase=0)
        _, _, _, _, self.base_pos, _ = self.p.getLinkState(self.robot, 0)
        _, _, _, _, self.camera_pos, _ = self.p.getLinkState(self.robot, 23)
        self.base_pos = np.array(self.base_pos)
        self.camera_pos = np.array(self.camera_pos)

        # Create viewers
        self.camera = Viewer(self.p, self.camera_pos, self.params["camera_look_pos"], 
                                fov=self.params["camera_fov"], 
                                near_pos=0.02, far_pos=7.0)
        
        if self.params.get("use_aux_camera", False):
            if "aux_camera_look_pos" not in self.params:
                self.params["aux_camera_look_pos"] = self.params["camera_look_pos"]
            if "aux_camera_fov" not in self.params:
                self.params["aux_camera_fov"] = self.params["camera_fov"]
            if "aux_image_size" not in self.params:
                self.params["aux_image_size"] = self.params["image_size"]
            self.aux_camera = Viewer(self.p, self.camera_pos, self.params["aux_camera_look_pos"], 
                                    fov=self.params["aux_camera_fov"],
                                    near_pos=0.02, far_pos=7.0)

        # Move arm to initial position
        self.move_joints_to_start()
        self.open_gripper()

        # Save state
        self.save_state()

    def save_state(self):
        self.saved_state = self.p.saveState()
        jointStates = self.p.getJointStates(self.robot, range(12,19))
        self.saved_joints = [state[0] for state in jointStates]

    def reset(self):
        if self.p: 
            self.p.restoreState(stateId=self.saved_state)
            for joint in range(7):
                self.p.setJointMotorControl2(self.robot,joint+12,self.p.POSITION_CONTROL,self.saved_joints[joint])
                self.p.setJointMotorControl2(self.robot,joint+12,self.p.VELOCITY_CONTROL,0)
        self.frames = []
    
    def spawn_walls(self, size=1.0):
        self.walls_id = self.p.loadURDF(URDF["walls"], globalScaling=size)

    def spawn_object(self, urdf, pos=[0,0,0], ori=None, scale=1.0):
        assert self.num_objects <= self.max_objects, "Maximum Number of Objects reached"
        if ori == None:
            ori = self.default_ori
        elif type(ori) == float or type(ori) == int:
            ori = self.p.getQuaternionFromEuler([0,0,ori])
        self.num_objects += 1
        self.object_dict[self.num_objects] = self.p.loadURDF(urdf, basePosition=pos, baseOrientation=ori, globalScaling=scale)
        return self.num_objects

    def get_object(self, object_id, relative=False):
        if relative:
            base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
            base_pos, base_ori = self.p.invertTransform(base_pos, base_ori)
            obj_pos, obj_ori = self.p.getBasePositionAndOrientation(self.object_dict[object_id])
            obj_pos, obj_ori = self.p.multiplyTransforms(base_pos, base_ori, obj_pos, obj_ori)
        else:
            obj_pos, obj_ori = self.p.getBasePositionAndOrientation(self.object_dict[object_id])
        return obj_pos, obj_ori

    def move_object(self, object_id, pos, ori=None, relative=False):
        if ori is None:
            ori = self.default_ori
        elif type(ori) == float or type(ori) == int:
            ori = self.p.getQuaternionFromEuler([0, 0, ori])
        elif len(ori) == 3:
            ori = self.p.getQuaternionFromEuler(ori)
        
        if relative:
            base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
            pos, ori = self.p.multiplyTransforms(base_pos, base_ori, pos, ori)

        self.p.resetBasePositionAndOrientation(self.object_dict[object_id], pos, ori)

    def remove_object(self, object_id):
        self.p.removeBody(self.object_dict[object_id])
        del self.object_dict[object_id]

    # def get_affordance(self, image=None):
    #     if image is None:
    #         image = self.render_camera()
    #     image = image.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,3).astype(np.uint8)
    #     output = GRASP_CLASSIFIER.predict(image)
    #     return output.item()

    def reset_robot(self, pos=[0, 0], yaw=0, left=0, right=0):
        self.set_base_pos_and_yaw(pos=pos, yaw=yaw)
        self.set_wheels_velocity(left, right)
        self.p.resetJointState(self.robot, 1, targetValue=0, targetVelocity=left)
        self.p.resetJointState(self.robot, 2, targetValue=0, targetVelocity=right)
        self.move_joints_to_start()

    def get_grasp_pos(self):
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        grasp_pos, grasp_ori = self.p.multiplyTransforms(base_pos, base_ori, self.params["gripper_ref_pos"], self.default_ori)
        return np.array(grasp_pos)

    def get_base_pos_and_yaw(self):
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        return np.array([base_pos[0], base_pos[1], self.p.getEulerFromQuaternion(base_ori)[2]])

    def set_base_pos_and_yaw(self, pos=np.array([0.0, 0.0]), yaw=0.0):
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        new_pos = [pos[0], pos[1], base_pos[2]]
        new_rot = self.p.getQuaternionFromEuler([0, 0, yaw])
        self.p.resetBasePositionAndOrientation(self.robot, new_pos, new_rot)

    def execute_grasp(self, pos, wrist_rotate=0.0):
        new_pos = self.params["pregrasp_pos"].copy()
        self.open_gripper(does_steps=False)
        self.move_ee(new_pos, wrist_rotate, steps=20, velocity_constrained=False)

        new_pos[:2] += pos
        self.move_ee(new_pos, wrist_rotate, steps=20, velocity_constrained=False)

        new_pos[2] = 0.1 #.02
        self.move_ee(new_pos, wrist_rotate, steps=30, velocity_constrained=True)
        self.close_gripper()

        new_pos[2] = 0.3
        self.move_ee(new_pos, wrist_rotate, steps=60, velocity_constrained=1.0)

    def set_wheels_velocity(self, left, right):
        self.p.setJointMotorControl2(self.robot, 1, self.p.VELOCITY_CONTROL, targetVelocity=left)
        self.p.setJointMotorControl2(self.robot, 2, self.p.VELOCITY_CONTROL, targetVelocity=right)

    def get_wheels_velocity(self):
        _, left, _, _ = self.p.getJointState(self.robot, 1)
        _, right, _, _ = self.p.getJointState(self.robot, 2)
        return np.array([left, right])

    def move_base(self, left, right):
        self.p.setJointMotorControl2(self.robot, 1, self.p.VELOCITY_CONTROL, targetVelocity=left, force=1e8)
        self.p.setJointMotorControl2(self.robot, 2, self.p.VELOCITY_CONTROL, targetVelocity=right, force=1e8)

        for i in range(45):
            self.step(i % 5 == 0)

        self.p.setJointMotorControl2(self.robot, 1, self.p.VELOCITY_CONTROL, targetVelocity=0, force=1e8)
        self.p.setJointMotorControl2(self.robot, 2, self.p.VELOCITY_CONTROL, targetVelocity=0, force=1e8)

        for i in range(55):
            self.step(i % 5 == 0)

    def move_ee(self, pos, wrist_rotate=0, steps=30, velocity_constrained=True):
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        pos, ori = self.p.multiplyTransforms(base_pos, base_ori, pos, self.params["down_quat"])
        jointStates = self.p.calculateInverseKinematics(self.robot, 16, pos, ori, maxNumIterations=70)[2:6]
        
        if velocity_constrained:
            if type(velocity_constrained) == bool:
                max_velocity = 5 
            else:
                max_velocity = velocity_constrained
        else:
            max_velocity = float("inf")
        for i in range(4):
            self.p.setJointMotorControl2(self.robot, i+12, self.p.POSITION_CONTROL, jointStates[i], maxVelocity=max_velocity)
        self.p.setJointMotorControl2(self.robot, 16, self.p.POSITION_CONTROL, wrist_rotate, maxVelocity=max_velocity)

        for i in range(steps):
            self.step(i % 5 == 0)

    def move_joints(self, joints, steps=69, velocity_constrained=True):
        max_velocity = 5 if velocity_constrained else float("inf")

        for i, j in enumerate(joints):
            self.p.setJointMotorControl2(self.robot, i+12, self.p.POSITION_CONTROL, j, maxVelocity=max_velocity)

        for i in range(steps):
            self.step(i % 5 == 0)

    def move_joints_to_start(self, steps=60):
        self.move_joints(self.params["start_joints"], steps=steps, velocity_constrained=False)

    def open_gripper(self, does_steps=True):
        self.p.setJointMotorControl2(self.robot, 17, self.p.POSITION_CONTROL, -.02)
        self.p.setJointMotorControl2(self.robot, 18, self.p.POSITION_CONTROL, .02)

        if does_steps:
            for i in range(33): #25):
                self.step(i % 5 == 0)

    def close_gripper(self, does_steps=True):
        self.p.setJointMotorControl2(self.robot, 17, self.p.POSITION_CONTROL, -0.001)
        self.p.setJointMotorControl2(self.robot, 18, self.p.POSITION_CONTROL, 0.001)

        if does_steps:
            for i in range(30): #25):
                self.step(i % 5 == 0)

    def step(self, record=False):
        if self.renders:
            time.sleep(self.step_duration)
        self.p.stepSimulation()
        
        if self.recording and record:
            camera_pos, camera_ori, _, _, _, _ = self.p.getLinkState(self.robot, 23)
            base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
            block_pos, block_ori = self.p.multiplyTransforms(base_pos, base_ori, self.params["camera_look_pos"], self.default_ori)
            self.camera.update(camera_pos, block_pos)
            frame = self.camera.get_image(width=240, height=240).flatten() * 1. / 255.
            self.frames.append(frame)

    def enable_recording(self):
        self.recording = True
        print("ENABLED RECORDING!!!")

    def get_frames(self):
        return self.frames

    def render_camera(self, use_aux=False):
        camera = self.aux_camera if use_aux else self.camera
        camera_look_pos = self.params["aux_camera_look_pos"] if use_aux else self.params["camera_look_pos"]

        camera_pos, camera_ori, _, _, _, _ = self.p.getLinkState(self.robot, 23)
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        look_pos, look_ori = self.p.multiplyTransforms(base_pos, base_ori, camera_look_pos, self.default_ori)
        camera.update(camera_pos, look_pos)

        if use_aux:
            image_width = self.params["aux_image_size"]
            image_height = self.params["aux_image_size"]
        else:
            image_width = self.params["image_size"]
            image_height = self.params["image_size"]
        image = camera.get_image(width=image_width, height=image_height)
        if self.grayscale:
            image = np.mean(image, axis=2).reshape((image_height, image_width, 1))
        return image.astype(np.uint8)
