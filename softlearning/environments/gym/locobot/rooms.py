import numpy as np

from .utils import *

def initialize_room(locobot_interface, name, room_params={}):
    if name == "simple":
        return SimpleRoom(locobot_interface, room_params)
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
            wall_size=5.0
        )
        defaults.update(params)
        super().__init__(interface, defaults)

        self._wall_size = self.params["wall_size"]
        self.obstacles_id.append(self.interface.spawn_object(URDF["walls"], scale=self._wall_size))
        
        self._num_objects = self.params["num_objects"]
        for i in range(self._num_objects):
            self.objects_id.append(self.interface.spawn_object(URDF[self.params["object_name"]], np.array([0.0, 0.0, 10 + i])))

    def reset(self):
        for i in range(self._num_objects):
            while True:
                x, y = np.random.uniform(-self._wall_size * 0.5, self._wall_size * 0.5, size=(2,))
                if not is_in_circle(x, y, 0, 0, 1.0):
                    break
            self.interface.move_object(self.objects_id[i], [x, y, 0.015])
    
    @property
    def num_objects(self):
        return self._num_objects

    @property
    def extent(self):
        return self._wall_size * 2.0