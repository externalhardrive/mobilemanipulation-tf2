import numpy as np

from .utils import *
from .objects import *

from softlearning.environments.helpers import random_point_in_circle

def initialize_room(locobot_interface, name, room_params={}):
    if name == "simple":
        return SimpleRoom(locobot_interface, room_params)
    elif name == "simple_obstacles":
        return SimpleRoomWithObstacles(locobot_interface, room_params)
    elif name == "medium":
        return MediumRoom(locobot_interface, room_params)
    elif name == "grasping":
        return GraspingRoom(locobot_interface, room_params)
    else:
        return NotImplementedError(f"no room has name {name}")

class Room:
    def __init__(self, interface, params):
        self.interface = interface
        self.objects_id = []
        self.obstacles_id = []
        self.params = params

    def reset(self, *args, **kwargs):
        return NotImplementedError

    @property
    def num_objects(self):
        return len(self.objects_id)

    @property
    def extent(self):
        """ Furthest distance from the origin considered to be in the room. """
        return 100

class SimpleRoom(Room):
    """ Simple room that has a wall and objects inside. """
    def __init__(self, interface, params):
        defaults = dict(
            num_objects=100, 
            object_name="greensquareball", 
            wall_size=5.0,
            no_spawn_radius=1.0,
        )
        defaults.update(params)
        super().__init__(interface, defaults)

        self._wall_size = self.params["wall_size"]
        self.obstacles_id.append(self.interface.spawn_object(URDF["walls"], scale=self._wall_size))

        self._no_spawn_radius = self.params["no_spawn_radius"]
        
        self._num_objects = self.params["num_objects"]
        for i in range(self._num_objects):
            self.objects_id.append(self.interface.spawn_object(URDF[self.params["object_name"]], np.array([0.0, 0.0, 10 + i])))

    def is_valid_spawn_loc(self, x, y, robot_pos=[0, 0]):
        return not is_in_circle(x, y, robot_pos[0], robot_pos[1], self._no_spawn_radius)

    def reset(self):
        for i in range(self._num_objects):
            while True:
                x, y = np.random.uniform(-self._wall_size * 0.5, self._wall_size * 0.5, size=(2,))
                if self.is_valid_spawn_loc(x, y):
                    break
            self.interface.move_object(self.objects_id[i], [x, y, 0.015])
    
    @property
    def num_objects(self):
        return self._num_objects

    @property
    def extent(self):
        return self._wall_size * 2.0

class GraspingRoom(Room):
    """ Room with objects spawn around specific location. """
    def __init__(self, interface, params):
        defaults = dict(
            min_objects=1,
            max_objects=10,
            object_name="greensquareball", 
            spawn_loc = [0, 0.42],
            spawn_radius = 0.10
        )
        defaults.update(params)
        super().__init__(interface, defaults)

        self._spawn_loc = self.params["spawn_loc"]
        self._spawn_radius = self.params["spawn_radius"]
        self.interface.change_floor_texture("wood")

        self._min_objects = self.params["min_objects"]
        self._max_objects = self.params["max_objects"]
        for i in range(self._max_objects):
            self.objects_id.append(self.interface.spawn_object(URDF[self.params["object_name"]], np.array([0.0, 0.0, 10 + i])))

    def is_valid_spawn_loc(self, x, y):
        return not is_in_circle(x, y, 0, 0, 0.2)

    def reset(self, num_objects=None):
        if not num_objects:
            num_objects = np.random.randint(self._min_objects, self._max_objects + 1)

        for i in range(num_objects):
            while True:
                dx, dy = random_point_in_circle(radius=(0, self._spawn_radius))
                x, y = self._spawn_loc[0] + dx, self._spawn_loc[1] + dy
                if self.is_valid_spawn_loc(x, y):
                    break
            self.interface.move_object(self.objects_id[i], [x, y, 0.015], ori=np.random.rand(4))
        
        for i in range(num_objects, self._max_objects):
            self.interface.move_object(self.objects_id[i], [self.extent + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.01])
    
    @property
    def num_objects(self):
        return self._max_objects

    @property
    def extent(self):
        return 10

class SimpleRoomWithObstacles(SimpleRoom):
    """ Simple room that has a wall and objects inside, with simple immovable obstacles (not randomly generated). """
    def __init__(self, interface, params):
        defaults = dict()
        super().__init__(interface, defaults)

        # don't spawn in 1m radius around the robot
        self.no_spawn_zones = []
        self.no_spawn_zones.append(lambda x, y: is_in_circle(x, y, 0, 0, 1.0))

        # add 4 rectangular pillars to the 4 corners
        c = self._wall_size / 4
        pillar_size = 0.5

        self.obstacles_id.append(self.interface.spawn_object(URDF["rectangular_pillar"], pos=[c, c, 0], scale=pillar_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["rectangular_pillar"], pos=[-c, c, 0], scale=pillar_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["rectangular_pillar"], pos=[c, -c, 0], scale=pillar_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["rectangular_pillar"], pos=[-c, -c, 0], scale=pillar_size))

        psh = pillar_size * 0.5
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,  c - psh,  c - psh,  c + psh,  c + psh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y, -c - psh,  c - psh, -c + psh,  c + psh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,  c - psh, -c - psh,  c + psh, -c + psh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y, -c - psh, -c - psh, -c + psh, -c + psh))

        # add 4 short boxes
        box_size = 0.25
        box_height = 0.2
        bsh = box_size * 0.5
        self.obstacles_id.append(self.interface.spawn_object(URDF["solid_box"], pos=[ c,  0, box_height - box_size], scale=box_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["solid_box"], pos=[-c,  0, box_height - box_size], scale=box_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["solid_box"], pos=[ 0,  c, box_height - box_size], scale=box_size))
        self.obstacles_id.append(self.interface.spawn_object(URDF["solid_box"], pos=[ 0, -c, box_height - box_size], scale=box_size))

        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,  c - bsh,     -bsh,  c + bsh,      bsh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y, -c - bsh,     -bsh, -c + bsh,      bsh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,     -bsh,  c - bsh,      bsh,  c + bsh))
        self.no_spawn_zones.append(lambda x, y: is_in_rect(x, y,     -bsh, -c - bsh,      bsh, -c + bsh))

    def is_valid_spawn_loc(self, x, y):
        for no_spawn in self.no_spawn_zones:
            if no_spawn(x, y):
                return False
        return True

class MediumRoom(Room):
    """ Simple room that has a wall and objects inside, with simple immovable obstacles (not randomly generated). """
    def __init__(self, interface, params):
        defaults = dict(
            num_objects=100, 
            object_name="greensquareball", 
            no_spawn_radius=1.0,
            wall_size=5.0,
        )
        defaults.update(params)
        super().__init__(interface, defaults)

        self._wall_size = self.params["wall_size"]
        self.wall_id = self.interface.spawn_object(URDF["walls_2"], scale=self._wall_size)

        self._num_objects = self.params["num_objects"]
        for i in range(self._num_objects):
            self.objects_id.append(self.interface.spawn_object(URDF[self.params["object_name"]], np.array([0.0, 0.0, 10 + i])))

        # don't spawn in 1m radius around the robot
        self.no_spawn_zones = []
        self._no_spawn_radius = self.params["no_spawn_radius"]
        self.no_spawn_zones.append(lambda x, y: is_in_circle(x, y, 0, 0, self._no_spawn_radius))

        self.interface.change_floor_texture("wood")

        boxes_config = [
            [[-2.259, 2.217, 0], 0, 0.219, "navy"],
            [[-1.776, -1.660, 0], 0, 0.383, "crate"],
            [[-1.12614, -2.08627, 0], 0, 0.321815, "crate"],
            [[-1.31922, 0.195723, 0], 0, 0.270704, "red"],
            [[1.39269, -2.35207, 0], 0, 0.577359, "marble"],
            [[0.33328, -0.75906, 0], 0, 0.233632, "navy"],
            [[2.08005, 1.24189, 0], 0, 0.361638, "crate"],
            [[-1.1331, 2.21413, 0], 0, 0.270704, "red"],
            [[-0.591208, 2.22201, 0], 0, 0.270704, "marble"],
            [[2.13627, 2.1474, 0], 0, 0.270704, "marble"],
            [[1.15959, 0.701976, 0], 0, 0.162524, "navy"],
            [[1.05702, 0.952707, 0], 0, 0.131984, "red"],
        ]

        self.static_objects = []
        for bc in boxes_config:
            self.static_objects.append(TexturedBox(self.interface, *bc[:2], bc[2]*2, texture_name=bc[3]))

    def is_valid_spawn_loc(self, x, y):
        for no_spawn in self.no_spawn_zones:
            if no_spawn(x, y):
                return False
        for so in self.static_objects:
            if so.is_point_inside(x, y):
                return False
        return True

    def reset(self):
        for i in range(self._num_objects):
            while True:
                x, y = np.random.uniform(-self._wall_size * 0.5, self._wall_size * 0.5, size=(2,))
                if self.is_valid_spawn_loc(x, y):
                    break
            self.interface.move_object(self.objects_id[i], [x, y, 0.02])
    
    @property
    def num_objects(self):
        return self._num_objects

    @property
    def extent(self):
        return self._wall_size * 2.0