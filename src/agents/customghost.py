from . import Ghost
from freegames import vector


class Blinky(Ghost.Ghost):
    color = "#FF0000"
    DOWN = vector(0, -10)
    UP = vector(0, 10)
    LEFT = vector(-10, 0)
    RIGHT = vector(10, 0)
    can_sense = False


class Pinky (Ghost.Ghost) :
    sense_reach = 2
    color = "#FC0FC0"

class Inky (Ghost. Ghost):
    color = "#00E6E6"


class Clyde (Ghost.Ghost):
    sense_reach = 4
    color = "#FF5B00"

