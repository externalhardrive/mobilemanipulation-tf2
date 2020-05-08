import os
import time
from os.path import expanduser
import sys
import pprint

import gym
import numpy as np
import pybullet
import pybullet_data
from gym import spaces
from gym.utils import seeding
from sklearn.linear_model import LinearRegression
from . import bullet_client

from .utils import *

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
    # CAMERA_LOOK_POS = np.array([0.5, 0., .2])
    # # BLOCK_POS = np.array([0.5, 0, 0.2])

    def __init__(self, **params):
        defaults = {
            "renders": False, # whether we use GUI mode or not
            "grayscale": False, # whether we render in grayscale
            "step_duration": 1/60, # when in render mode, how long in seconds does each step take
            "start_joints": np.array([0., -0.6, 1.3, 0.5, 1.6]), # joint values for the neutral start position
            "pregrasp_pos": np.array([0.42, 0, 0.3]), # local coord for the end-effector pos to go to before grasping
            "down_quat": np.array([0.0, 0.7071067811865475, 0.0, 0.7071067811865476]), # quaternion for gripper to point downwards
            "camera_look_pos": np.array([0.5, 0., .2]), # local pos that the camera looks at
            "image_size": 100, # size of the image that the camera renders
            "camera_fov": 60 # FOV of the camera
        }
        defaults.update(params)
        self.params = defaults

        print()
        print("LocobotInterface params:")
        pprint.pprint(dict(
            self=self,
            **self.params,
        ))
        print()

        self.renders = self.params["renders"]
        self.grayscale = self.params["grayscale"]
        self.step_duration = self.params["step_duration"]

        # set up pybullet simulation
        if self.renders:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            print("LOCOBOT: STARTED RENDER MODE")
        else:
            self.p = bullet_client.BulletClient()

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.8)
        self.default_ori = self.p.getQuaternionFromEuler([0,0,0])

        # set up texture storage
        self.texture_name_to_id = {}

        # Load the base plane
        self.plane_id = self.p.loadURDF("plane_transparent.urdf")
        # self.plane_id = self.p.loadURDF("plane100.urdf", useMaximalCoordinates=True)

        # Load robot
        self.robot_urdf = URDF["locobot"]
        self.robot = self.p.loadURDF(self.robot_urdf, useFixedBase=0)
        _, _, _, _, self.base_pos, _ = self.p.getLinkState(self.robot, 0)
        _, _, _, _, self.camera_pos, _ = self.p.getLinkState(self.robot, 23)
        self.base_pos = np.array(self.base_pos)
        self.camera_pos = np.array(self.camera_pos)

        # Create viewers
        self.camera = Viewer(self.p, self.camera_pos, self.params["camera_look_pos"], 
                                fov=self.params["camera_fov"], 
                                near_pos=0.02, far_pos=7.0)
        
        # create the second auxilary camera if specified
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
    

    # ----- TEXTURES METHOD -----

    def change_floor_texture(self, texture_name):
        self.p.changeVisualShape(self.plane_id, -1, textureUniqueId=self.load_texture(texture_name))

    def load_texture(self, texture_name):
        if texture_name in self.texture_name_to_id:
            return self.texture_name_to_id[texture_name]
        else:
            tex_id = self.p.loadTexture(TEXTURE[texture_name])
            self.texture_name_to_id[texture_name] = tex_id
            return tex_id

    # ----- END TEXTURES METHOD -----
    

    # ----- OBJECTS METHOD -----

    def spawn_walls(self, size=1.0):
        self.walls_id = self.p.loadURDF(URDF["walls"], globalScaling=size)

    def spawn_object(self, urdf, pos=[0,0,0], ori=None, scale=1.0):
        """ Spawn the given object
        Args:
            object_id: the ID of the object
            pos: (3,) vector of the object position
            ori: None - default orientation. 
                 float or int - yaw rotation.
                 (3,) vec - euler rotation (rpy)
                 (4,) vec - quaternion
            scale: float scale of the oject
        """
        if ori is None:
            ori = self.default_ori
        elif type(ori) == float or type(ori) == int:
            ori = self.p.getQuaternionFromEuler([0, 0, ori])
        elif len(ori) == 3:
            ori = self.p.getQuaternionFromEuler(ori)
        return self.p.loadURDF(urdf, basePosition=pos, baseOrientation=ori, globalScaling=scale)

    def get_object(self, object_id, relative=False):
        if relative:
            base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
            base_pos, base_ori = self.p.invertTransform(base_pos, base_ori)
            obj_pos, obj_ori = self.p.getBasePositionAndOrientation(object_id)
            obj_pos, obj_ori = self.p.multiplyTransforms(base_pos, base_ori, obj_pos, obj_ori)
        else:
            obj_pos, obj_ori = self.p.getBasePositionAndOrientation(object_id)
        return obj_pos, obj_ori

    def move_object(self, object_id, pos, ori=None, relative=False):
        """ Move the given object
        Args:
            object_id: the ID of the object
            pos: (3,) vector of the object position
            ori: None - default orientation. 
                 float or int - yaw rotation.
                 (3,) vec - euler rotation (rpy)
                 (4,) vec - quaternion
            relative: if True, interpret args as local coordinates relative to the robot's frame of ref 
        """
        if ori is None:
            ori = self.default_ori
        elif type(ori) == float or type(ori) == int:
            ori = self.p.getQuaternionFromEuler([0, 0, ori])
        elif len(ori) == 3:
            ori = self.p.getQuaternionFromEuler(ori)
        
        if relative:
            base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
            pos, ori = self.p.multiplyTransforms(base_pos, base_ori, pos, ori)

        self.p.resetBasePositionAndOrientation(object_id, pos, ori)

    def remove_object(self, object_id):
        self.p.removeBody(object_id)

    # ----- END OBJECTS METHOD -----



    # ----- BASE METHODS -----

    def reset_robot(self, pos=[0, 0], yaw=0, left=0, right=0):
        """ Reset the robot's position and move the arm back to start.
        Args:
            pos: (2,) vector. Assume that the robot is on the floor.
            yaw: float. Rotation of the robot around the z-axis.
            left: left wheel velocity.
            right: right wheel velocity.
        """
        self.set_base_pos_and_yaw(pos=pos, yaw=yaw)
        self.set_wheels_velocity(left, right)
        self.p.resetJointState(self.robot, 1, targetValue=0, targetVelocity=left)
        self.p.resetJointState(self.robot, 2, targetValue=0, targetVelocity=right)
        self.move_joints_to_start()

    def get_base_pos_and_yaw(self):
        """ Get the base position and yaw. (x, y, yaw). """
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        return np.array([base_pos[0], base_pos[1], self.p.getEulerFromQuaternion(base_ori)[2]])

    def get_base_pos(self):
        """ Get the base position (x, y). """
        base_pos, _ = self.p.getBasePositionAndOrientation(self.robot)
        return np.array([base_pos[0], base_pos[1]])

    def set_base_pos_and_yaw(self, pos=np.array([0.0, 0.0]), yaw=0.0):
        """ Set the base's position (x, y) and yaw. """
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        new_pos = [pos[0], pos[1], base_pos[2]]
        new_rot = self.p.getQuaternionFromEuler([0, 0, yaw])
        self.p.resetBasePositionAndOrientation(self.robot, new_pos, new_rot)

    def set_wheels_velocity(self, left, right):
        self.p.setJointMotorControl2(self.robot, 1, self.p.VELOCITY_CONTROL, targetVelocity=left)
        self.p.setJointMotorControl2(self.robot, 2, self.p.VELOCITY_CONTROL, targetVelocity=right)

    def get_wheels_velocity(self):
        _, left, _, _ = self.p.getJointState(self.robot, 1)
        _, right, _, _ = self.p.getJointState(self.robot, 2)
        return np.array([left, right])
    
    def move_base(self, left, right):
        """ Move the base by some amount. """
        self.p.setJointMotorControl2(self.robot, 1, self.p.VELOCITY_CONTROL, targetVelocity=left, force=1e8)
        self.p.setJointMotorControl2(self.robot, 2, self.p.VELOCITY_CONTROL, targetVelocity=right, force=1e8)

        self.do_steps(45)

        self.p.setJointMotorControl2(self.robot, 1, self.p.VELOCITY_CONTROL, targetVelocity=0, force=1e8)
        self.p.setJointMotorControl2(self.robot, 2, self.p.VELOCITY_CONTROL, targetVelocity=0, force=1e8)

        self.do_steps(55)

    # ----- END BASE METHODS -----



    # ----- ARM METHODS -----

    def execute_grasp(self, pos, wrist_rotate=0.0):
        """ Do a predetermined single grasp action by doing the following:
            1. Move end-effector to the pregrasp_pos.
            2. Offset the effector x,y postion by the given pos.
            3. Move end-effector downwards and grasp.
            4. Move end-effector upwards
        Args:
            pos: (x,y) location of the grasp with origin at the pregrasp_pos, local to the robot
            wrist_rotate: wrist rotation in radians
        """
        new_pos = self.params["pregrasp_pos"].copy()
        self.open_gripper(steps=0)
        self.move_ee(new_pos, wrist_rotate, steps=20)

        new_pos[:2] += pos
        self.move_ee(new_pos, wrist_rotate, steps=20)

        new_pos[2] = 0.1 #.02
        self.move_ee(new_pos, wrist_rotate, steps=30, max_velocity=5.0)
        self.close_gripper()

        new_pos[2] = 0.3
        self.move_ee(new_pos, wrist_rotate, steps=60, max_velocity=1.0)

    def move_ee(self, pos, wrist_rotate=0, steps=30, max_velocity=float("inf"), ik_steps=70):
        """ Move the end-effector to the given pos, pointing down.
        Args:
            pos: (3,) vector local coordinate for the desired end effector position.
            wrist_rotate: rotation of the wrist in radians. 0 is the gripper closing from the sides.
            steps: how many simulation steps to do.
            max_velocity: the maximum velocity of the joints..
            ik_steps: how many IK steps to calculate the final joint values.
        """
        base_pos, base_ori = self.p.getBasePositionAndOrientation(self.robot)
        pos, ori = self.p.multiplyTransforms(base_pos, base_ori, pos, self.params["down_quat"])
        jointStates = self.p.calculateInverseKinematics(self.robot, 16, pos, ori, maxNumIterations=ik_steps)[2:6]
        
        for i in range(4):
            self.p.setJointMotorControl2(self.robot, i+12, self.p.POSITION_CONTROL, jointStates[i], maxVelocity=max_velocity)
        self.p.setJointMotorControl2(self.robot, 16, self.p.POSITION_CONTROL, wrist_rotate, maxVelocity=max_velocity)

        self.do_steps(steps)

    def move_joints(self, joints, steps=69, max_velocity=float("inf")):
        """ Move the arms joints to the given joints values.
        Args:
            pos: (5,) vector of the 5 joint pos
            steps: how many simulation steps to do
            max_velocity: the maximum velocity of the joints
        """
        for i, j in enumerate(joints):
            self.p.setJointMotorControl2(self.robot, i+12, self.p.POSITION_CONTROL, j, maxVelocity=max_velocity)

        self.do_steps(steps)

    def move_joints_to_start(self, steps=60, max_velocity=float("inf")):
        """ Move the arms joints to the start_joints position
        Args:
            pos: (5,) vector of the 5 joint pos
            steps: how many simulation steps to do
        """
        self.move_joints(self.params["start_joints"], steps=steps, max_velocity=max_velocity)

    def open_gripper(self, steps=30):
        """ Open the gripper in steps simulation steps. """
        self.p.setJointMotorControl2(self.robot, 17, self.p.POSITION_CONTROL, -.02)
        self.p.setJointMotorControl2(self.robot, 18, self.p.POSITION_CONTROL, .02)

        self.do_steps(steps)

    def close_gripper(self, steps=30):
        """ Close the gripper in steps simulation steps. """
        self.p.setJointMotorControl2(self.robot, 17, self.p.POSITION_CONTROL, -0.001)
        self.p.setJointMotorControl2(self.robot, 18, self.p.POSITION_CONTROL, 0.001)

        self.do_steps(steps)

    # ----- END ARM METHODS -----



    # ----- MISC METHODS -----

    def step(self):
        """ Do a single simulation step. If in GUI mode, then this takes step_duration seconds. """
        if self.renders:
            time.sleep(self.step_duration)
        self.p.stepSimulation()
        
    def do_steps(self, num_steps):
        """ Do num_steps simulation steps. If in GUI mode, then this takes num_steps * step_duration seconds. """
        for _ in range(num_steps):
            self.step()

    def render_camera(self, use_aux=False):
        """ Renders the scene
        Args:
            use_aux: determines whether this renders using the main camera or the auxilary camera.
        Returns:
            (height, width, channel) uint8 array
        """
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

    # ----- END MISC METHODS -----
