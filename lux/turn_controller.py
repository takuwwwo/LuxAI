from typing import Optional

from lux.game_constants import GAME_CONSTANTS
from lux.game import Game


class TurnController:
    def __init__(self, game: Game):
        self.game = game

        self.turn = self.game.turn
        self.day_length = GAME_CONSTANTS['PARAMETERS']['DAY_LENGTH']
        self.night_length = GAME_CONSTANTS['PARAMETERS']['NIGHT_LENGTH']
        self.cycle_length = self.day_length + self.night_length
        self.max_turn = GAME_CONSTANTS['PARAMETERS']['MAX_DAYS']

    def is_day(self, turn: Optional[int] = None) -> bool:
        day_in_turn = self.get_day_in_turn(turn)

        if 0 <= day_in_turn < self.day_length:
            return True
        else:
            return False

    def is_night(self, turn: Optional[int] = None) -> bool:
        day_in_turn = self.get_day_in_turn(turn)
        if self.day_length <= day_in_turn < self.cycle_length:
            return True
        else:
            return False

    def next_night(self, turn: Optional[int] = None) -> int:
        if self.is_night(turn):
            return 0

        day_in_turn = self.get_day_in_turn(turn)
        return self.day_length - day_in_turn

    def next_day(self, turn: Optional[int] = None) -> int:
        if self.is_day(turn):
            return 0

        day_in_turn = self.get_day_in_turn(turn)
        return self.day_length - day_in_turn

    def get_day_in_turn(self, turn: Optional[int] = None):
        return self.get_turn(turn) % self.cycle_length

    def get_number_cycle(self, turn: Optional[int]):
        return self.get_turn(turn) // self.cycle_length

    def get_turn(self, turn: Optional[int] = None) -> int:
        if turn:
            return turn
        else:
            return self.turn

    def count_night(self, start_turn: int, end_turn: int) -> int:
        """
        count the number of nights from start_turn to end_turn
        Note that end_turn itself is not counted.
        :param start_turn:
        :param end_turn:
        :return:
        """
        start_turn = max(start_turn, 0)
        end_turn = min(end_turn, self.max_turn)
        if start_turn >= end_turn:
            return 0

        start_cycle_no = self.get_number_cycle(start_turn)
        end_cycle_no = self.get_number_cycle(end_turn)
        num_cycles = end_cycle_no - start_cycle_no

        start_day_in_turn = self.get_day_in_turn(start_turn)
        end_day_in_turn = self.get_day_in_turn(end_turn)

        num_nights = num_cycles * self.night_length + max(end_day_in_turn - self.day_length, 0) - max(
            start_day_in_turn - self.day_length, 0)
        return num_nights
