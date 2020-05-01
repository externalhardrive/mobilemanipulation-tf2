import numpy as np

from .utils import *

class StaticObject:
    def __init__(self, interface, pos, rot, scale):
        self.interface = interface
        self.pos = pos
        self.rot = rot
        self.scale = scale

    def is_point_inside(self, x, y):
        return False

class TexturedBox(StaticObject):
    def __init__(self, *args, texture_name=None):
        super().__init__(*args)
        self.id = self.interface.spawn_object(URDF["textured_box"], pos=self.pos, ori=0, scale=self.scale)
        self.interface.p.changeVisualShape(self.id, -1, textureUniqueId=self.interface.load_texture(texture_name))

    def is_point_inside(self, x, y):
        return is_in_rect(x, y,  
            self.pos[0] - self.scale * 0.5,
            self.pos[1] - self.scale * 0.5,
            self.pos[0] + self.scale * 0.5,
            self.pos[1] + self.scale * 0.5,)