from typing import List, Tuple
import json
import random
from dataclasses import dataclass

from torch.utils.data import Dataset
import numpy as np

from lux.game_map import DIRECTIONS
from lux.game_objects import Player
from lux.game import Game
from lux.turn_controller import TurnController
from agent_constants import MAP_SIZE, UnitAction, CitytileAction
from agent_ import convert_state


@dataclass
class DatasetOutput:
    state_array: np.array
    action_array: np.array
    target_array: np.array
    city_action_array: np.array
    city_target_array: np.array
    masking: np.array
    agent_label: int


class LuxDataset(Dataset):
    def __init__(self, obses_list: List[str], agent_labels_list: List[int]):
        self.obses_list = obses_list
        self.agent_labels_list = agent_labels_list
        self.map_size = MAP_SIZE

    def __len__(self):
        return len(self.obses_list)

    def __getitem__(self, idx) -> DatasetOutput:
        filepath, turn, player_id = self.obses_list[idx]
        with open(filepath) as f:
            json_load = json.load(f)

        actions = json_load['steps'][turn + 1][player_id]['action']
        if actions is None:
            actions = []
        obs = json_load['steps'][turn][0]['observation']
        obs['player'] = player_id

        agent_label = self.agent_labels_list[idx]
        width, height = obs['width'], obs['height']
        turn = obs['step']
        if turn == 0:
            messages = obs['updates'][2:]
        else:
            messages = obs['updates']

        game_state = Game()
        game_state.initialize2(messages, player_id, turn, width, height)
        turn_controller = TurnController(game_state)
        player = game_state.players[player_id]
        opponent = game_state.players[1 - player_id]

        state_array, masking = convert_state(game_state, player, opponent, turn_controller, self.map_size)

        # Encode unit actions
        # default is MOVE_CENTER
        action_array = np.ones((self.map_size, self.map_size)).astype(np.long) * UnitAction.MOVE_CENTER
        target_array = np.zeros((self.map_size, self.map_size)).astype(np.float32)
        unit_dict = dict()
        for unit in player.units:
            unit_dict[unit.id] = unit
            if unit.can_act():
                pos_x, pos_y = self.get_pos_in_state(unit.pos.x, unit.pos.y, width, height)
                target_array[pos_x][pos_y] = 1.0
        for action in actions:
            unit_id, action_label = self.label_action(action, player)

            try:
                unit = unit_dict[unit_id]
            except KeyError:
                continue
            if unit.can_act():
                pos_x, pos_y = self.get_pos_in_state(unit.pos.x, unit.pos.y, width, height)
                action_array[pos_x, pos_y] = action_label  # overwrite (isn't cared the duplicate case)

        # encode citytile actions
        # default is DO_NOTHING
        city_action_array = np.ones((self.map_size, self.map_size)).astype(np.long) * CitytileAction.DO_NOTHING

        city_target_array = np.zeros((self.map_size, self.map_size)).astype(np.float32)
        city_tile_dict = dict()
        for city in player.cities.values():
            for city_tile in city.citytiles:
                pos_x, pos_y = self.get_pos_in_state(city_tile.pos.x, city_tile.pos.y, width, height)
                city_tile_dict[(city_tile.pos.x, city_tile.pos.y)] = city_tile
                if city_tile.can_act():
                    city_target_array[pos_x][pos_y] = 1.0
        for action in actions:
            pos, city_tile_label = self.label_city_action(action)
            try:
                city_tile = city_tile_dict[pos]
            except KeyError:
                continue

            if city_tile.can_act():
                pos_x, pos_y = self.get_pos_in_state(city_tile.pos.x, city_tile.pos.y, width, height)
                city_action_array[pos_x, pos_y] = city_tile_label

        rand_value = random.randint(0, 3)  # for data augmentation
        if rand_value == 0:
            return DatasetOutput(state_array, action_array, target_array, city_action_array, city_target_array, masking,
                                 agent_label)
        elif rand_value == 1:  # rotate 90 degree
            state_array, action_array, target_array, city_action_array, city_target_array = \
                self.rotate90(state_array, action_array, target_array, city_action_array, city_target_array)
            return DatasetOutput(state_array, action_array, target_array, city_action_array, city_target_array, masking,
                                 agent_label)
        elif rand_value == 2:  # rotate 180 degree
            state_array, action_array, target_array, city_action_array, city_target_array = \
                self.rotate180(state_array, action_array, target_array, city_action_array, city_target_array)
            return DatasetOutput(state_array, action_array, target_array, city_action_array, city_target_array, masking,
                                 agent_label)
        else:  # rotate 270 degree
            state_array, action_array, target_array, city_action_array, city_target_array = \
                self.rotate270(state_array, action_array, target_array, city_action_array, city_target_array)
            return DatasetOutput(state_array, action_array, target_array, city_action_array, city_target_array, masking,
                                 agent_label)

    @staticmethod
    def label_action(action: str, player: Player) -> Tuple[str, int]:
        str_list = action.split(' ')
        unit_id = str_list[1]
        if str_list[0] == 'm':
            label = {'n': UnitAction.MOVE_NORTH, 'w': UnitAction.MOVE_WEST, 's': UnitAction.MOVE_SOUTH,
                     'e': UnitAction.MOVE_EAST, 'c': UnitAction.MOVE_CENTER}[str_list[2]]
        elif str_list[0] == 'bcity':
            label = UnitAction.BUILD_CITY
        elif str_list[0] == 't':
            target_id = str_list[2]
            units = player.units

            pos, target_pos = None, None
            for unit in units:
                if unit.id == unit_id:
                    pos = unit.pos
                if unit.id == target_id:
                    target_pos = unit.pos
            if pos is None or target_pos is None:
                label = UnitAction.MOVE_CENTER
            else:
                if pos.translate(DIRECTIONS.NORTH, 1) == target_pos:
                    label = UnitAction.TRANSFER_NORTH
                elif pos.translate(DIRECTIONS.WEST, 1) == target_pos:
                    label = UnitAction.TRANSFER_WEST
                elif pos.translate(DIRECTIONS.SOUTH, 1) == target_pos:
                    label = UnitAction.TRANSFER_SOUTH
                elif pos.translate(DIRECTIONS.EAST, 1) == target_pos:
                    label = UnitAction.TRANSFER_EAST
                else:
                    label = UnitAction.MOVE_CENTER
        else:
            label = UnitAction.MOVE_CENTER   # others actions are set as MOVE_CENTER
        return unit_id, label

    @staticmethod
    def label_city_action(action) -> Tuple[Tuple[int, int], CitytileAction]:
        str_list = action.split(' ')
        if str_list[0] == 'r':  # research
            pos_x, pos_y = str_list[1], str_list[2]
            label = CitytileAction.RESEARCH
        elif str_list[0] == 'bw' or str_list[0] == 'bc':  # build worker or build cart
            pos_x, pos_y = str_list[1], str_list[2]
            label = CitytileAction.BUILD_WORKER
        else:
            pos_x, pos_y = -1, -1
            label = CitytileAction.DO_NOTHING
        return (int(pos_x), int(pos_y)), label

    def get_pos_in_state(self, xx: int, yy: int, width: int, height: int) -> Tuple[int, int]:
        """ state内での座標を返す
        """
        pos_in_state = xx + (self.map_size - width) // 2, yy + (self.map_size - height) // 2
        return pos_in_state

    @staticmethod
    def rotate90(state_array, action_array, target_array, city_action_array, city_target_array):
        state_array = np.rot90(state_array, 1, axes=(-1, -2)).copy()
        action_array = np.rot90(action_array, 1, axes=(-1, -2)).copy()
        target_array = np.rot90(target_array, 1, axes=(-1, -2)).copy()
        city_action_array = np.rot90(city_action_array, axes=(-1, -2)).copy()
        city_target_array = np.rot90(city_target_array, axes=(-1, -2)).copy()

        returned_action_array = action_array.copy()
        returned_action_array[action_array == 0] = 1
        returned_action_array[action_array == 1] = 2
        returned_action_array[action_array == 2] = 3
        returned_action_array[action_array == 3] = 0
        returned_action_array[action_array == 6] = 7
        returned_action_array[action_array == 7] = 8
        returned_action_array[action_array == 8] = 9
        returned_action_array[action_array == 9] = 6

        return state_array, returned_action_array, target_array, city_action_array, city_target_array

    @staticmethod
    def rotate180(state_array, action_array, target_array, city_action_array, city_target_array):
        state_array = np.rot90(state_array, 2, axes=(-1, -2)).copy()
        action_array = np.rot90(action_array, 2, axes=(-1, -2)).copy()
        target_array = np.rot90(target_array, 2, axes=(-1, -2)).copy()
        city_action_array = np.rot90(city_action_array, 2, axes=(-1, -2)).copy()
        city_target_array = np.rot90(city_target_array, 2, axes=(-1, -2)).copy()

        returned_action_array = action_array.copy()
        returned_action_array[action_array == 0] = 2
        returned_action_array[action_array == 1] = 3
        returned_action_array[action_array == 2] = 0
        returned_action_array[action_array == 3] = 1
        returned_action_array[action_array == 6] = 8
        returned_action_array[action_array == 7] = 9
        returned_action_array[action_array == 8] = 6
        returned_action_array[action_array == 9] = 7

        return state_array, returned_action_array, target_array, city_action_array, city_target_array

    @staticmethod
    def rotate270(state_array, action_array, target_array, city_action_array, city_target_array):
        state_array = np.rot90(state_array, 3, axes=(-1, -2)).copy()
        action_array = np.rot90(action_array, 3, axes=(-1, -2)).copy()
        target_array = np.rot90(target_array, 3, axes=(-1, -2)).copy()
        city_action_array = np.rot90(city_action_array, 3, axes=(-1, -2)).copy()
        city_target_array = np.rot90(city_target_array, 3, axes=(-1, -2)).copy()

        returned_action_array = action_array.copy()
        returned_action_array[action_array == 0] = 3
        returned_action_array[action_array == 1] = 0
        returned_action_array[action_array == 2] = 1
        returned_action_array[action_array == 3] = 2
        returned_action_array[action_array == 6] = 9
        returned_action_array[action_array == 7] = 6
        returned_action_array[action_array == 8] = 7
        returned_action_array[action_array == 9] = 8

        return state_array, returned_action_array, target_array, city_action_array, city_target_array
