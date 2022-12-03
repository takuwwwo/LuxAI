from typing import List, Tuple

import torch
import numpy as np

from lux.game import Game
from lux.game_map import RESOURCE_TYPES, DIRECTIONS
from lux.game_objects import Player
from lux.game_constants import GAME_CONSTANTS
from agent_constants import STATE_CHANNELS
from lux.turn_controller import TurnController
from model.policy_network import PolicyNetwork


def convert_state(game_state: Game, player: Player, opponent: Player, turn_controller: TurnController,
                  state_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    channel_size = STATE_CHANNELS
    width, height = game_state.map_width, game_state.map_height

    def get_pos_in_state(xx: int, yy: int):
        pos_in_state = xx + (state_size - width) // 2, yy + (state_size - height) // 2
        return pos_in_state

    state = np.zeros((channel_size, state_size, state_size)).astype(np.float32)
    first_x, first_y = get_pos_in_state(0, 0)
    last_x, last_y = get_pos_in_state(width, height)

    # channel[0]: the number of units
    for unit in player.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        state[0, x, y] += 1.

    # channel[1:4]: the amount of each resource of units
    for unit in player.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        state[1, x, y] = unit.cargo.wood / 100.
        state[2, x, y] = unit.cargo.coal / 100.
        state[3, x, y] = unit.cargo.uranium / 100.

    # channel[4]: cooldown of units
    state[4, first_x:last_x, first_y:last_y] = GAME_CONSTANTS["PARAMETERS"]['UNIT_ACTION_COOLDOWN']['WORKER'] * 2
    for unit in player.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        state[4, x, y,] = min(state[4, x, y], unit.cooldown) / 10.0

    # channel[5]: number of opponent units
    for unit in opponent.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        state[5, x, y] += 1.

    # channel[6:9]: amount of each resource of units
    for unit in opponent.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        state[6, x, y] = unit.cargo.wood / 100.
        state[7, x, y] = unit.cargo.coal / 100.
        state[8, x, y] = unit.cargo.uranium / 100.

    # channel[9]: cooldown of opponent units
    state[9, first_x:last_x, first_y:last_y] = GAME_CONSTANTS["PARAMETERS"]['UNIT_ACTION_COOLDOWN']['WORKER'] * 2
    for unit in player.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        state[9, x, y] = min(state[9, x, y], unit.cooldown) / 10.0

    # channel[10:13]: wood, coal, uranium in each cell
    for x_ in range(width):
        for y_ in range(height):
            cell = game_state.map.get_cell(x_, y_)
            if not cell.has_resource():
                continue

            x, y = get_pos_in_state(x_, y_)
            if cell.resource.type == RESOURCE_TYPES.WOOD:
                state[10, x, y] = cell.resource.amount / 500.
            elif cell.resource.type == RESOURCE_TYPES.COAL:
                state[11, x, y] = cell.resource.amount / 500.
            elif cell.resource.type == RESOURCE_TYPES.URANIUM:
                state[12, x, y] = cell.resource.amount / 500.

    # channel[13]: my research point
    state[13, first_x:last_x, first_y:last_y] = player.research_points / 200.

    # channel[14]: opponent research point
    state[14, first_x:last_x, first_y:last_y] = opponent.research_points / 200.

    # channel[15]: current turn / max turn
    state[15, first_x:last_x, first_y:last_y] = turn_controller.get_turn() / turn_controller.max_turn

    # channel[16]: turn in one cycles
    state[16, first_x:last_x, first_y:last_y] = turn_controller.get_day_in_turn() / turn_controller.cycle_length

    # channel[17]: day or night
    if turn_controller.is_day():
        state[17, first_x:last_x, first_y:last_y] = 1.0

    # channel[18]: size of map
    state[18, first_x:last_x, first_y:last_y] = width / 32.

    # channel[19]: number of my citytile
    # channel[20]: amount of my citytile's fuel
    # channel[21]: cooldown of my citytile
    state[21, first_x:last_x, first_y:last_y] = GAME_CONSTANTS["PARAMETERS"]['CITY_ACTION_COOLDOWN']
    for city in player.cities.values():
        for citytile in city.citytiles:
            x, y = get_pos_in_state(citytile.pos.x, citytile.pos.y)
            state[19, x, y] = 1.
            state[20, x, y] = city.fuel / 1000.
            state[21, x, y] = citytile.cooldown / 10.

    # channel[22]: number of opponent citytile
    # channel[23]: amount of opponent citytile's fuel
    # channel[24]: cooldown of opponent citytile
    for city in opponent.cities.values():
        for citytile in city.citytiles:
            x, y = get_pos_in_state(citytile.pos.x, citytile.pos.y)
            state[22, x, y] = 1.
            state[23, x, y] = city.fuel / 1000.
            state[24, x, y] = citytile.cooldown / 10.

    # channel[25]: whether unit can act or not
    state[25, first_x:last_x, first_y:last_y] = 0.0
    for unit in player.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        if unit.cooldown == 0:
            state[25, x, y] = 1.0

    # channel[26]: whether opponent unit can act or not
    state[26, first_x:last_x, first_y:last_y] = 0.0
    for unit in opponent.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        if unit.cooldown == 0:
            state[26, x, y] = 1.0

    # channel[27]: whether citytile can build or not
    state[27, first_x:last_x, first_y:last_y] = 0.0
    for unit in player.units:
        if unit.is_worker() and unit.can_build(game_map=game_state.map):
            x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
            state[27, x, y] = 1.0

    # channel[28]: whether opponent citytile can build or not
    state[28, first_x:last_x, first_y:last_y] = 0.0
    for unit in opponent.units:
        if unit.is_worker() and unit.can_build(game_map=game_state.map):
            x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
            state[28, x, y] = 1.0

    # channel[29]: units' weighted resource
    for unit in player.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        state[29, x, y] = unit.cargo.wood * GAME_CONSTANTS["PARAMETERS"]['RESOURCE_TO_FUEL_RATE']['WOOD'] \
                          + unit.cargo.coal * GAME_CONSTANTS["PARAMETERS"]['RESOURCE_TO_FUEL_RATE']['COAL'] \
                          + unit.cargo.uranium * GAME_CONSTANTS["PARAMETERS"]['RESOURCE_TO_FUEL_RATE']['URANIUM']
        state[29, x, y] = state[29, x, y] / (100. * 40.)

    # channel[30]: opponent units' weighted resource
    for unit in opponent.units:
        x, y = get_pos_in_state(unit.pos.x, unit.pos.y)
        state[30, x, y] = unit.cargo.wood * GAME_CONSTANTS["PARAMETERS"]['RESOURCE_TO_FUEL_RATE']['WOOD'] \
                          + unit.cargo.coal * GAME_CONSTANTS["PARAMETERS"]['RESOURCE_TO_FUEL_RATE']['COAL'] \
                          + unit.cargo.uranium * GAME_CONSTANTS["PARAMETERS"]['RESOURCE_TO_FUEL_RATE']['URANIUM']
        state[30, x, y] = state[30, x, y] / (100. * 40.)

    # channel[31]: number of turns to night
    state[31, first_x:last_x, first_y:last_y] = turn_controller.next_night() / 30.

    # channel[32]: fuel burn in city
    for city in player.cities.values():
        for citytile in city.citytiles:
            x, y = get_pos_in_state(citytile.pos.x, citytile.pos.y)
            state[32, x, y] = city.light_upkeep / 200.
    return state, state[18, :, :] <= 0.0


class Agent:
    def __init__(self, policy_net: PolicyNetwork, device, research_th: float = 0.0, research_turn: int = 10):
        self.game_state = None
        self.policy_net = policy_net
        self.player_id = None
        self.policy_device = device
        self.exploration_turn = None
        self.research_th = research_th
        self.research_turn = research_turn
        self.map_size = 32

    def __call__(self, observation, configuration) -> List[str]:
        ### Do not edit ###
        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state._update(observation["updates"][2:])
            self.game_state.id = observation.player

            player: Player = self.game_state.players[observation.player]
            opponent: Player = self.game_state.players[1 - observation.player]
            self.player_id = observation.player
        else:
            if self.game_state is None:
                raise ValueError
            player: Player = self.game_state.players[self.player_id]
            opponent: Player = self.game_state.players[1 - self.player_id]

            self.game_state._update(observation["updates"])

        game_state = self.game_state
        turn_controller = TurnController(self.game_state)
        state_array, masking_array = convert_state(self.game_state, player, opponent, turn_controller)
        target_array = state_array[25, :, :]

        state = torch.FloatTensor(state_array).to(self.policy_device).unsqueeze(0)  # make sample batchsize 1
        masking = torch.BoolTensor(masking_array).to(self.policy_device).unsqueeze(0)  # make sample batchsize 1
        target = torch.FloatTensor(target_array).to(self.policy_device).unsqueeze(0)  # make sample batchsize 1
        (actions, action_probs, _), (citytile_actions, citytile_action_probs, _) = self.policy_net.act(state,
                                                                                                       masking, target)

        actions = actions[0].to('cpu').detach().numpy()  # actions.shape == (H, W)
        action_probs = action_probs[0].to('cpu').detach().numpy()  # action_probs.shape == (H, W, num_actions)
        if citytile_action_probs is not None:
            citytile_action_probs = citytile_action_probs[0].to('cpu').detach().numpy()

        result_actions = self.make_actions_list(game_state, player, action_probs, actions, citytile_action_probs)
        return result_actions

    def make_actions_list(self, game_state: Game, player: Player, action_probs: np.ndarray, actions: np.ndarray,
                          citytile_action_probs: np.ndarray):
        state_size = self.map_size
        width, height = game_state.map_width, game_state.map_height

        def get_pos_in_state(xx: int, yy: int):
            pos_in_state = xx + (state_size - width) // 2, yy + (state_size - height) // 2
            return pos_in_state

        def in_city(pos):
            try:
                city = game_state.map.get_cell_by_pos(pos).citytile
                return city is not None and city.team == game_state.id
            except:
                return False

        action_list = []
        # city actions
        unit_count = len(player.units)
        build_worker_candidates = []
        for city in player.cities.values():
            for citytile in city.citytiles:
                if citytile.can_act():
                    x, y = get_pos_in_state(citytile.pos.x, citytile.pos.y)
                    prob = citytile_action_probs[x, y, 1]
                    prob2 = citytile_action_probs[x, y, 0]
                    build_worker_candidates.append((citytile, prob, prob2))
        build_worker_candidates.sort(key=lambda x: x[1], reverse=True)  # rearrange by higher prob
        for citytile, prob, prob2 in build_worker_candidates:
            if unit_count < player.city_tile_count:
                action_list.append(citytile.build_worker())
                unit_count += 1
            elif not player.researched_uranium():
                if prob2 < self.research_th and game_state.turn <= self.research_turn:
                    continue
                action_list.append(citytile.research())
                player.research_points += 1

        pos_unit_dict = dict()
        for worker in player.units:
            pos_unit_dict[worker.pos.x * 1000 + worker.pos.y] = worker

        # Unit Actions
        dest = []
        source_dest_pair = []
        for worker in player.units:
            if not worker.can_act():
                dest.append(worker.pos)

        for worker in player.units:
            if not worker.can_act():
                continue

            x, y = get_pos_in_state(worker.pos.x, worker.pos.y)
            policy = action_probs[x, y, :]
            prioritized_action = actions[x, y]
            action_candidates = [prioritized_action] + list(np.argsort(policy))[::-1]
            for action in action_candidates:
                if 0 <= action <= 5:
                    dir = DIRECTIONS.get_dir(action)
                    if dir is None:  # action is build city
                        if worker.pos not in dest and not in_city(worker.pos):
                            # if there is no worker and city, build city
                            action_list.append(worker.build_city())
                            dest.append(worker.pos)
                            break
                    else:
                        next_pos = worker.pos.translate(dir, 1)
                        if next_pos not in dest or in_city(next_pos):
                            # avoid going to same cell from same cell
                            if (worker.pos, next_pos) in source_dest_pair:
                                continue

                            # if there is no worker, or citytile, do move
                            action_list.append(worker.move(dir))
                            dest.append(next_pos)
                            if worker.pos != next_pos:
                                source_dest_pair.append((worker.pos, next_pos))
                            break
                else:  # transfer
                    dir = DIRECTIONS.get_dir(action - 6)
                    if worker.pos in dest:
                        continue

                    target_pos = worker.pos.translate(dir, 1)
                    target_pos_encoded = target_pos.x * 1000 + target_pos.y
                    if target_pos_encoded in pos_unit_dict:
                        target_unit = pos_unit_dict[target_pos_encoded]
                        target_id = target_unit.id
                        if worker.cargo.uranium > 0:
                            action_list.append(worker.transfer(target_id, RESOURCE_TYPES.URANIUM, 2000))
                        elif worker.cargo.coal > 0:
                            action_list.append(worker.transfer(target_id, RESOURCE_TYPES.COAL, 2000))
                        elif worker.cargo.wood > 0:
                            action_list.append(worker.transfer(target_id, RESOURCE_TYPES.WOOD, 2000))
                        else:
                            continue
                    dest.append(worker.pos)
                    break

        return action_list

    def get_unit_pos(self, player: Player):
        if len(player.units) == 0:
            return None

        worker = player.units[0]
        return worker.pos
