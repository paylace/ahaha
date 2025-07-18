# settings.py

class SimulationSettings:
    """시뮬레이션 전체에 사용되는 상수와 설정을 관리하는 클래스입니다."""

    SPATIAL_HASH_GRID_SIZE = 5 # 5x5 크기의 타일을 하나의 큰 격자로 묶음
    
    # 환경 설정
    WIDTH = 40
    HEIGHT = 40
    MAX_FOOD_PER_TILE = 100
    FOOD_REGEN_RATE = 10
    BARRIER_UPDATE_INTERVAL = 15
    BARRIER_CHANGE_PROB = 0.1

    # 초기 개체군 설정
    INITIAL_PREY_COUNT = 200
    INITIAL_PREDATOR_COUNT = 15
    MAX_TURNS = 300

    # === [최종] 유전 및 종 분화 관련 상수 ===
    ALLELE_MUTATION_RATE = 0.005
    CONTINUOUS_MUTATION_RATE = 0.01
    MUTATION_STRENGTH = 0.1
    SPECIATION_THRESHOLD = 0.2
    LINEAGE_MISMATCH_PENALTY = 0.8 # 계통 불일치 시 번식 성공률 감소율 (80% 감소)
    # ======================================

    # 유전적 유사성 가중치
    GENETIC_SIMILARITY_WEIGHTS = {
        'size': 0.15, 'muscle_mass': 0.15, 'color': 0.2,
        'reproduction_cycle': 0.15, 'offspring_count_base': 0.1,
        'resource_preference': 0.15, 'lifespan_potential': 0.1,
        'mate_pref_color': 0.05, 'mate_pref_lineage': 0.05,
        'foraging_strategy': 0.05, 'hunting_strategy': 0.05,
        'learning_rate': 0.05
    }
    GENETIC_SIMILARITY_THRESHOLD = 0.7
    GENETIC_SIMILARITY_PENALTY_FACTOR = 0.5
    
    # 나머지 상수들
    BASE_MOVEMENT_SPEED_INDIVIDUAL = 1
    BASE_ENERGY_CONSUMPTION_INDIVIDUAL = 5
    MAX_ENERGY_INDIVIDUAL = 100
    MAX_AGE_INDIVIDUAL = 50
    CARCASS_FOOD_VALUE_INDIVIDUAL = 10
    BASE_MOVEMENT_SPEED_PREDATOR = 3
    BASE_ENERGY_CONSUMPTION_PREY = 3
    MAX_ENERGY_PREY = 50
    MAX_AGE_PREY = 10
    CARCASS_FOOD_VALUE_PREY = 5
    PREY_REPRO_ENERGY_COST_FACTOR = 0.3
    PREY_MIN_REPRO_ENERGY_FACTOR = 0.5
    BASE_ENERGY_CONSUMPTION_PREDATOR = 5
    MAX_ENERGY_PREDATOR = 150
    MAX_AGE_PREDATOR = 15
    CARCASS_FOOD_VALUE_PREDATOR = 15
    PREDATOR_HUNTING_RANGE = 5
    PREDATOR_REPRO_ENERGY_COST_FACTOR = 0.4
    PREDATOR_MIN_REPRO_ENERGY_FACTOR = 0.6
    LIFESPAN_POTENTIAL_AGE_BONUS_FACTOR = 0.1
    LIFESPAN_POTENTIAL_ENERGY_COST_FACTOR = 0.5
    REPRODUCTION_CYCLE_MISMATCH_PENALTY = 0.5
    MIN_REPRODUCTION_CHANCE = 0.1
    SIZE_EFFECT_ON_MOVEMENT = 1
    MUSCLE_EFFECT_ON_MOVEMENT = 2
    SIZE_EFFECT_ON_ENERGY_CONSUMPTION = 2
    MUSCLE_EFFECT_ON_ENERGY_CONSUMPTION = 1
    PREY_SIZE_EFFECT_ON_ESCAPE = 0.05
    PREY_MUSCLE_EFFECT_ON_ESCAPE = 0.05
    PREY_CAMOUFLAGE_EFFECT_ON_ESCAPE = 0.1
    PREDATOR_SIZE_PENALTY_ON_ESCAPE = 0.03
    PREDATOR_MUSCLE_PENALTY_ON_ESCAPE = 0.04
    PRED_SIZE_EFFECT_ON_HUNT = 0.05
    PRED_MUSCLE_EFFECT_ON_HUNT = 0.04
    PRED_CAMOUFLAGE_EFFECT_ON_HUNT = 0.1
    PREY_SIZE_PENALTY_ON_HUNT = 0.05
    PREY_MUSCLE_PENALTY_ON_HUNT = 0.06
    PREY_CAMOUFLAGE_PENALTY_ON_HUNT = 0.1
    PREY_FOOD_INTAKE_BASE_CAPACITY = 5
    PREY_FOOD_INTAKE_SIZE_BONUS = 2
    PREY_RESOURCE_PREF_BONUS = 8
    PREY_REPRO_FOOD_BONUS_FACTOR = 0.8
    PREDATOR_REPRO_ENERGY_BONUS_FACTOR = 0.5
    MATE_PREF_COLOR_TYPES = ['Red', 'Blue', 'Purple', 'Any']
    MATE_PREF_LINEAGE_TYPES = ['Same', 'Different', 'Any']
    MATE_PREF_COLOR_MATCH_BONUS = 0.2
    MATE_PREF_LINEAGE_MATCH_BONUS = 0.3
    MATE_PREF_DIST_PENALTY_FACTOR = 0.05
    MIN_REPRODUCTION_COMPATIBILITY = 0.5
    TRAIT_DIVERGENCE_PENALTY_RATE = 1.0
    FORAGING_STRATEGIES = ['normal', 'wide_range', 'stealth']
    PREY_WIDE_RANGE_MOVE_BONUS = 1
    PREY_WIDE_RANGE_ENERGY_COST_MULTIPLIER = 1.2
    PREY_STEALTH_ESCAPE_BONUS = 0.15
    PREY_STEALTH_ENERGY_COST_MULTIPLIER = 0.8
    HUNTING_STRATEGIES = ['pursuit', 'ambush']
    PREDATOR_AMBUSH_HUNT_SUCCESS_BONUS = 0.2
    PREDATOR_AMBUSH_MIN_HUNTING_RANGE = 2
    PREDATOR_AMBUSH_MOVEMENT_REDUCTION = 0.5
    LEARNING_RATE_DECAY = 0.05
    LEARNING_BONUS_FACTOR = 0.2
    LEARNING_PENALTY_FACTOR = 0.05
    LEARNED_SCORE_MIN = -1.0
    LEARNED_SCORE_MAX = 1.0
