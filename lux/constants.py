from typing import Optional

class Constants:
    class INPUT_CONSTANTS:
        RESEARCH_POINTS = "rp"
        RESOURCES = "r"
        UNITS = "u"
        CITY = "c"
        CITY_TILES = "ct"
        ROADS = "ccd"
        DONE = "D_DONE"
    class DIRECTIONS:
        NORTH = "n"
        WEST = "w"
        SOUTH = "s"
        EAST = "e"
        CENTER = "c"

        @classmethod
        def get_dir(cls, index: int) -> Optional[str]:
            if index == 0:
                return cls.NORTH
            elif index == 1:
                return cls.WEST
            elif index == 2:
                return cls.SOUTH
            elif index == 3:
                return cls.EAST
            elif index == 4:
                return cls.CENTER
            else:
                return None

    class UNIT_TYPES:
        WORKER = 0
        CART = 1
    class RESOURCE_TYPES:
        WOOD = "wood"
        URANIUM = "uranium"
        COAL = "coal"
