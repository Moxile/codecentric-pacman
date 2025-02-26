from . import Ghost
from freegames import vector


class Blinky(Ghost.Ghost):
    color = "red"
    DOWN = vector(0, -10)
    UP = vector(0, 10)
    LEFT = vector(-10, 0)
    RIGHT = vector(10, 0)
    can_sense = False


class Pinky (Ghost.Ghost) :
    color = "pink"

class Inky (Ghost. Ghost):
    color = "teal"


class Clyde (Ghost.Ghost):
    color = "orange"

