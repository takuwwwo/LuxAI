from typing import Dict

from .constants import Constants
from .game_map import Position, Cell, GameMap, Resource
from .game_constants import GAME_CONSTANTS

UNIT_TYPES = Constants.UNIT_TYPES
DIRECTIONS = Constants.DIRECTIONS
RESOURCE_TYPES = Constants.RESOURCE_TYPES


class Player:
    def __init__(self, team):
        self.team = team
        self.research_points = 0
        self.units: list[Unit] = []
        self.cities: Dict[str, City] = {}
        self.city_tile_count = 0

    def researched_coal(self) -> bool:
        return self.research_points >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"]

    def researched_uranium(self) -> bool:
        return self.research_points >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]


class City:
    def __init__(self, teamid, cityid, fuel, light_upkeep):
        self.cityid = cityid
        self.team = teamid
        self.fuel = fuel
        self.citytiles: list[CityTile] = []
        self.light_upkeep = light_upkeep

    def _add_city_tile(self, x, y, cooldown):
        ct = CityTile(self.team, self.cityid, x, y, cooldown)
        self.citytiles.append(ct)
        return ct

    def get_light_upkeep(self):
        return self.light_upkeep


class CityTile:
    def __init__(self, teamid, cityid, x, y, cooldown):
        self.cityid = cityid
        self.team = teamid
        self.pos = Position(x, y)
        self.cooldown = cooldown

    def can_act(self) -> bool:
        """
        Whether or not this unit can research or build
        """
        return self.cooldown < 1

    def research(self) -> str:
        """
        returns command to ask this tile to research this turn
        """
        return "r {} {}".format(self.pos.x, self.pos.y)

    def build_worker(self) -> str:
        """
        returns command to ask this tile to build a worker this turn
        """
        return "bw {} {}".format(self.pos.x, self.pos.y)

    def build_cart(self) -> str:
        """
        returns command to ask this tile to build a cart this turn
        """
        return "bc {} {}".format(self.pos.x, self.pos.y)


class Cargo:
    fuel_rate = {
        'wood': GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['WOOD'],
        'coal': GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['COAL'],
        'uranium': GAME_CONSTANTS['PARAMETERS']['RESOURCE_TO_FUEL_RATE']['URANIUM']
    }

    def __init__(self):
        self.wood = 0
        self.coal = 0
        self.uranium = 0

    def __str__(self) -> str:
        return f"Cargo | Wood: {self.wood}, Coal: {self.coal}, Uranium: {self.uranium}"

    def get_amount(self) -> int:
        return self.wood + self.coal + self.uranium

    def lose_resource(self, lose_fuel: int):
        while lose_fuel > 0:
            if self.get_amount() == 0:
                break

            if self.wood > 0:
                lost_wood = self._calc_lost_resource_unit(self.wood, self.fuel_rate['wood'], lose_fuel)
                self.wood -= lost_wood
                lose_fuel = lost_wood * self.fuel_rate['wood']
            elif self.coal > 0:
                lost_coal = self._calc_lost_resource_unit(self.coal, self.fuel_rate['coal'], lose_fuel)
                self.coal -= lost_coal
                lose_fuel -= lost_coal * self.fuel_rate['coal']
            elif self.uranium > 0:
                lost_uranium = self._calc_lost_resource_unit(self.uranium, self.fuel_rate['uranium'], lose_fuel)
                self.uranium -= lost_uranium
                lose_fuel -= lost_uranium * self.fuel_rate['uranium']

        if lose_fuel > 0:
            raise ValueError

    def get_fuel_value(self) -> float:
        return self.wood * self.fuel_rate['wood'] + self.coal * self.fuel_rate['coal'] + self.uranium * self.fuel_rate[
            'uranium']

    def lose_resource_multiple_times(self, lose_fuel: int, num_of_reduce: int):
        target_resource_list = [self.wood, self.coal, self.uranium]

        # 1回で資源がどれだけ減るか
        resource_reduce_list = [
            self.reduced_amount_one_turn(self.fuel_rate['wood'], lose_fuel),
            self.reduced_amount_one_turn(self.fuel_rate['coal'], lose_fuel),
            self.reduced_amount_one_turn(self.fuel_rate['uranium'], lose_fuel)
        ]
        for target_resource, reduced_amount in zip(target_resource_list, resource_reduce_list):
            if num_of_reduce <= 0:
                break

            # 今のリソースタイプで足りる場合
            if target_resource >= reduced_amount:
                target_resource -= reduced_amount * num_of_reduce
                num_of_reduce = 0
            else:
                num_of_reduce_tmp = target_resource / reduced_amount  # 今の資源で何回減らせるか
                num_of_reduce -= num_of_reduce_tmp
                target_resource -= reduced_amount * num_of_reduce_tmp  # 減らす分を引く
                self.lose_resource(lose_fuel)  # 余った分を適切に引く
                num_of_reduce -= 1

        # 資源を消費しきれなかった場合はエラー
        if num_of_reduce > 0:
            raise ValueError

    @staticmethod
    def reduced_amount_one_turn(lose_fuel: int, fool_rate: int) -> int:
        """
        1回で消費する資源量
        :param lose_fuel:
        :param fool_rate:
        :return:
        """
        return (lose_fuel + fool_rate - 1) // fool_rate

    @staticmethod
    def _calc_lost_resource_unit(max_unit: int, fuel_rate: int, lose_fuel: int):
        lost_fuel = min(lose_fuel, max_unit * fuel_rate)
        return (lost_fuel + fuel_rate - 1) // fuel_rate


class Unit:
    def __init__(self, teamid, u_type, unitid, x, y, cooldown, wood, coal, uranium):
        self.pos = Position(x, y)
        self.team = teamid
        self.id = unitid
        self.type = u_type
        self.cooldown = cooldown
        self.cargo = Cargo()
        self.cargo.wood = wood
        self.cargo.coal = coal
        self.cargo.uranium = uranium

    def is_worker(self) -> bool:
        return self.type == UNIT_TYPES.WORKER

    def is_cart(self) -> bool:
        return self.type == UNIT_TYPES.CART

    def get_cargo_space_left(self):
        """
        get cargo space left in this unit
        """
        spaceused = self.cargo.wood + self.cargo.coal + self.cargo.uranium
        if self.type == UNIT_TYPES.WORKER:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] - spaceused
        else:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] - spaceused

    def can_build(self, game_map) -> bool:
        """
        whether or not the unit can build where it is right now
        """
        cell = game_map.get_cell_by_pos(self.pos)
        if not cell.has_resource() and self.can_act() and (self.cargo.wood + self.cargo.coal + self.cargo.uranium) >= \
                GAME_CONSTANTS["PARAMETERS"]["CITY_BUILD_COST"]:
            return True
        return False

    def can_act(self) -> bool:
        """
        whether or not the unit can move or not. This does not check for potential collisions into other units or enemy cities
        """
        return self.cooldown < 1

    def move(self, dir) -> str:
        """
        return the command to move unit in the given direction
        """
        return "m {} {}".format(self.id, dir)

    def transfer(self, dest_id, resourceType, amount) -> str:
        """
        return the command to transfer a resource from a source unit to a destination unit as specified by their ids
        """
        return "t {} {} {} {}".format(self.id, dest_id, resourceType, amount)

    def build_city(self) -> str:
        """
        return the command to build a city right under the worker
        """
        return "bcity {}".format(self.id)

    def pillage(self) -> str:
        """
        return the command to pillage whatever is underneath the worker
        """
        return "p {}".format(self.id)
