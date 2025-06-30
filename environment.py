# environment.py
import random
import collections
from settings import SimulationSettings

class Environment:
    """시뮬레이션 환경을 정의하고 관리하는 클래스입니다."""
    def __init__(self, width, height, max_food_per_tile, food_regen_rate):
        self.width = width
        self.height = height
        self.max_food_per_tile = max_food_per_tile
        self.food_regen_rate = food_regen_rate
        self.grid, self.barrier_map = self._initialize_grid()
        self.occupancy_map = collections.defaultdict(set)
        self.simulation_ref = None
        self.current_turn = 0

    def _initialize_grid(self):
        self.environment_colors = ['Red', 'Blue', 'Purple']
        grid = {}
        barrier_map = {}
        barrier_column_x = self.width // 2
        left_color = 'Red'
        right_color = 'Blue'
        pref_prob = 0.7
        for x in range(self.width):
            for y in range(self.height):
                color_pool = [c for c in self.environment_colors if c != (left_color if x < barrier_column_x else right_color)]
                env_color = random.choices([(left_color if x < barrier_column_x else right_color)] + color_pool, weights=[pref_prob] + [(1-pref_prob)/2]*2)[0]
                grid[(x, y)] = {'food_amount': self.max_food_per_tile / 2, 'color': env_color}
                barrier_map[(x, y)] = (x == barrier_column_x)
        return grid, barrier_map

    def update_food(self):
        for pos in self.grid:
            self.grid[pos]['food_amount'] = min(self.max_food_per_tile, self.grid[pos]['food_amount'] + self.food_regen_rate)

    def get_food_amount(self, x, y):
        return self.grid.get((x, y), {'food_amount': 0})['food_amount']

    def consume_food(self, x, y, amount):
        current_food = self.get_food_amount(x, y)
        consumed = min(current_food, amount)
        if (x, y) in self.grid:
            self.grid[(x, y)]['food_amount'] -= consumed
        return consumed

    def add_food_from_carcass(self, x, y, amount):
        if (x, y) in self.grid:
            self.grid[(x, y)]['food_amount'] = min(self.max_food_per_tile, self.grid[(x, y)]['food_amount'] + amount)

    def get_environment_color(self, x, y):
        return self.grid.get((x, y), {'color': None})['color']

    def is_valid_position(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_position_empty(self, x, y):
        return not self.occupancy_map.get((x,y))

    def is_barrier(self, x, y):
        return self.barrier_map.get((x, y), False)

    def add_individual_to_occupancy_map(self, ind):
        self.occupancy_map[(ind.x, ind.y)].add(ind.id)

    def remove_individual_from_occupancy_map(self, ind):
        loc_set = self.occupancy_map.get((ind.x, ind.y))
        if loc_set and ind.id in loc_set:
            loc_set.remove(ind.id)

    def update_barriers(self, turn):
        interval = SimulationSettings.BARRIER_UPDATE_INTERVAL
        prob = SimulationSettings.BARRIER_CHANGE_PROB
        if turn > 0 and turn % interval == 0:
            for (x, y), is_barrier in list(self.barrier_map.items()):
                if is_barrier and random.random() < prob:
                    self.barrier_map[(x, y)] = False
            
            empty_tiles = [(x, y) for x in range(self.width) for y in range(self.height) if not self.is_barrier(x, y) and self.is_position_empty(x, y)]
            for x, y in empty_tiles:
                if random.random() < prob:
                    self.barrier_map[(x, y)] = True
            
            if not any(self.barrier_map.values()) and empty_tiles:
                self.barrier_map[random.choice(empty_tiles)] = True
