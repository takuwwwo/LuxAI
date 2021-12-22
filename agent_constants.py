from enum import IntEnum


MAP_SIZE = 32
UNIT_ACTIONS = 10
CITYTILE_ACTIONS = 3
STATE_CHANNELS = 33


class UnitAction(IntEnum):
    MOVE_NORTH = 0
    MOVE_WEST = 1
    MOVE_SOUTH = 2
    MOVE_EAST = 3
    MOVE_CENTER = 4
    BUILD_CITY = 5
    TRANSFER_NORTH = 6
    TRANSFER_WEST = 7
    TRANSFER_SOUTH = 8
    TRANSFER_EAST = 9


class CitytileAction(IntEnum):
    RESEARCH = 0
    BUILD_WORKER = 1
    DO_NOTHING = 2
