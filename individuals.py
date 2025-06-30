# individuals.py
import random
import collections
from settings import SimulationSettings
import numpy as np # NumPy import 추가
from utils import calculate_similarity_jit # 새로 만든 JIT 함수 import

class Individual:
    _next_id = 0
    def __init__(self, env, x, y, age=0, energy=0, parent_id=None, birth_turn=None):
        self.id = Individual._next_id; Individual._next_id += 1
        self.env = env; self.x = x; self.y = y; self.age = age; self.energy = energy
        self.is_alive = True; self.parent_id = parent_id
        self.birth_turn = birth_turn if birth_turn is not None else env.current_turn
        self.env.add_individual_to_occupancy_map(self)
        self._initialize_traits(); self.genotype = self._generate_genotype_from_phenotype()
        self._initialize_base_stats()
      
    def _calculate_genetic_similarity(self, other):
        """
        두 개체의 유전자형을 비교하여 유전적 유사성 점수(0.0~1.0)를 계산합니다.
        핵심 계산은 Numba로 컴파일된 외부 함수를 호출하여 속도를 높입니다.
        """
        total_sim, total_weight = 0.0, 0.0
        for trait, weight in SimulationSettings.GENETIC_SIMILARITY_WEIGHTS.items():
            if not hasattr(self, trait) or not hasattr(other, trait) or self.genotype.get(trait) is None: continue
            
            total_weight += weight
            gene1, gene2 = self.genotype.get(trait), other.genotype.get(trait)
            trait_sim = 0.0
            
            if isinstance(gene1, (list, tuple)):
                alleles1_list = (gene1[0] + gene1[1]) if isinstance(gene1[0], list) else gene1
                alleles2_list = (gene2[0] + gene2[1]) if isinstance(gene2[0], list) else gene2
                
                # JIT 함수에 전달하기 위해 NumPy 배열로 변환
                # 빈 문자열 ''이 들어올 경우를 대비해 if c를 추가하여 안정성 확보
                alleles1_np = np.array([ord(c[0]) for c in alleles1_list if c], dtype=np.int64)
                alleles2_np = np.array([ord(c[0]) for c in alleles2_list if c], dtype=np.int64)
                
                # 두 배열이 모두 비어있지 않을 때만 JIT 함수를 호출
                if alleles1_np.size > 0 and alleles2_np.size > 0:
                    trait_sim = calculate_similarity_jit(alleles1_np, alleles2_np)
                # (중복 호출 라인이 삭제되었습니다)

            elif trait == 'learning_rate':
                trait_sim = 1.0 - (abs(float(gene1[0]) - float(gene2[0])) / 0.9)
            
            total_sim += trait_sim * weight
        
        return total_sim / total_weight if total_weight > 0 else 1.0
    
    def _initialize_traits(self):
        self.size = self._generate_random_trait_value('size'); self.muscle_mass = self._generate_random_trait_value('muscle_mass')
        self.color = self._generate_random_trait_value('color'); self.lifespan_potential = self._generate_random_trait_value('lifespan_potential')
        self.reproduction_cycle = self._generate_random_trait_value('reproduction_cycle', self.__class__.__name__)
        self.offspring_count_base = self._generate_random_trait_value('offspring_count_base'); self.lineage = 'A'
        self.resource_preference = self._generate_random_trait_value('resource_preference'); self.mate_pref_color = self._generate_random_trait_value('mate_pref_color')
        self.mate_pref_lineage = self._generate_random_trait_value('mate_pref_lineage')
        if isinstance(self, Prey): self.foraging_strategy = self._generate_random_trait_value('foraging_strategy'); self.hunting_strategy = None
        elif isinstance(self, Predator): self.hunting_strategy = self._generate_random_trait_value('hunting_strategy'); self.foraging_strategy = None
        self.learning_rate = self._generate_random_trait_value('learning_rate')
        self.learned_resource_pref_scores = self._initialize_learned_scores('resource_preference', self.env.environment_colors)
        self.learned_mating_pref_scores = self._initialize_learned_scores('mate_pref_color', SimulationSettings.MATE_PREF_COLOR_TYPES)
        if isinstance(self, Prey): self.learned_foraging_strategy_scores = self._initialize_learned_scores('foraging_strategy', SimulationSettings.FORAGING_STRATEGIES)
        elif isinstance(self, Predator): self.learned_hunting_strategy_scores = self._initialize_learned_scores('hunting_strategy', SimulationSettings.HUNTING_STRATEGIES)

    def _initialize_base_stats(self):
        self.base_movement_speed = SimulationSettings.BASE_MOVEMENT_SPEED_INDIVIDUAL; self.base_energy_consumption = SimulationSettings.BASE_ENERGY_CONSUMPTION_INDIVIDUAL
        self.max_energy = SimulationSettings.MAX_ENERGY_INDIVIDUAL; self.max_age = SimulationSettings.MAX_AGE_INDIVIDUAL
        self.carcass_food_value = SimulationSettings.CARCASS_FOOD_VALUE_INDIVIDUAL

    def _initialize_learned_scores(self, pref_type, options):
        scores = {opt: 0.0 for opt in options}; initial_pref = getattr(self, pref_type)
        if initial_pref in scores: scores[initial_pref] = 0.1
        return scores

    def _generate_random_trait_value(self, trait_name, species_type=None):
        if trait_name == 'size': return random.randint(1, 5)
        if trait_name == 'muscle_mass': return random.randint(1, 5)
        if trait_name == 'color': return random.choice(['Red', 'Blue', 'Purple'])
        if trait_name == 'reproduction_cycle': return random.choice([1, 2, 4] if species_type == 'Prey' else [3, 4, 5])
        if trait_name == 'resource_preference': return random.choice(['Red', 'Blue', 'Purple'])
        if trait_name == 'lifespan_potential': return random.randint(1, 5)
        if trait_name == 'offspring_count_base': return random.randint(1, 3)
        if trait_name == 'mate_pref_color': return random.choice(SimulationSettings.MATE_PREF_COLOR_TYPES)
        if trait_name == 'mate_pref_lineage': return random.choice(SimulationSettings.MATE_PREF_LINEAGE_TYPES)
        if trait_name == 'foraging_strategy': return random.choice(SimulationSettings.FORAGING_STRATEGIES)
        if trait_name == 'hunting_strategy': return random.choice(SimulationSettings.HUNTING_STRATEGIES)
        if trait_name == 'learning_rate': return random.uniform(0.1, 1.0)
        return 0

    def _generate_genotype_from_phenotype(self):
        genotype = {}; color_map = {'Red': ['A', 'A'], 'Blue': ['B', 'B'], 'Purple': ['A', 'B']}
        genotype['color'] = color_map[self.color]
        def generate_alleles(value, p, s):
            pool = [p] * (value - 1) + [s] * (5 - value); random.shuffle(pool)
            return (sorted(pool[:2]), sorted(pool[2:]))
        genotype['size'] = generate_alleles(self.size, 'S1', 's1')
        genotype['muscle_mass'] = generate_alleles(self.muscle_mass, 'M1', 'm1')
        repro_map = {1:['C','C'], 2:['C','c'], 4:['c','c'], 3:['C','C'], 5:['c','c']}
        genotype['reproduction_cycle'] = repro_map.get(self.reproduction_cycle, ['C','c'])
        offspring_map = {3:['O','O'], 2:['O','o'], 1:['o','o']}
        genotype['offspring_count_base'] = offspring_map[self.offspring_count_base]
        for trait in ['lineage', 'resource_preference', 'lifespan_potential', 'mate_pref_color', 'mate_pref_lineage', 'foraging_strategy', 'hunting_strategy', 'learning_rate']:
            if hasattr(self, trait) and getattr(self, trait) is not None: genotype[trait] = [str(getattr(self, trait))] * 2
        return genotype

    def calculate_phenotype_from_genotype(self, genotype_data):
        phenotype = {}; color_alleles = genotype_data.get('color', [])
        if 'A' in color_alleles and 'B' in color_alleles: phenotype['color'] = 'Purple'
        elif 'A' in color_alleles: phenotype['color'] = 'Red'
        else: phenotype['color'] = 'Blue'
        def calculate_dominant_trait(alleles_pair): return sum(1 for allele in alleles_pair[0] if allele.isupper()) + sum(1 for allele in alleles_pair[1] if allele.isupper())
        phenotype['size'] = 1 + calculate_dominant_trait(genotype_data.get('size', ([],[])))
        phenotype['muscle_mass'] = 1 + calculate_dominant_trait(genotype_data.get('muscle_mass', ([],[])))
        repro_alleles = genotype_data.get('reproduction_cycle', [])
        if 'C' in repro_alleles and 'c' in repro_alleles: phenotype['reproduction_cycle'] = 2 if isinstance(self, Prey) else 4
        elif 'C' in repro_alleles: phenotype['reproduction_cycle'] = 1 if isinstance(self, Prey) else 3
        else: phenotype['reproduction_cycle'] = 4 if isinstance(self, Prey) else 5
        offspring_alleles = genotype_data.get('offspring_count_base', [])
        if 'O' in offspring_alleles and 'o' in offspring_alleles: phenotype['offspring_count_base'] = 2
        elif 'O' in offspring_alleles: phenotype['offspring_count_base'] = 3
        else: phenotype['offspring_count_base'] = 1
        for trait in ['lineage', 'resource_preference', 'lifespan_potential', 'mate_pref_color', 'mate_pref_lineage', 'foraging_strategy', 'hunting_strategy', 'learning_rate']:
            if trait in genotype_data:
                value = genotype_data[trait][0]
                if trait in ['lifespan_potential']: phenotype[trait] = int(value)
                elif trait in ['learning_rate']: phenotype[trait] = float(value)
                else: phenotype[trait] = value
        return phenotype

    def _apply_mutation(self, genotype):
        new_genotype = {k: list(v) if isinstance(v, tuple) else v[:] for k, v in genotype.items()}
        for trait, alleles in new_genotype.items():
            if trait in ['size', 'muscle_mass']:
                for i in range(2):
                    for j in range(2):
                        if random.random() < SimulationSettings.ALLELE_MUTATION_RATE:
                            alleles[i][j] = alleles[i][j].lower() if alleles[i][j].isupper() else alleles[i][j].upper()
                new_genotype[trait] = (sorted(alleles[0]), sorted(alleles[1]))
            elif trait == 'learning_rate':
                if random.random() < SimulationSettings.CONTINUOUS_MUTATION_RATE:
                    new_rate = max(0.01, min(1.0, float(alleles[0]) + random.gauss(0, SimulationSettings.MUTATION_STRENGTH)))
                    new_genotype[trait] = [str(new_rate)] * 2
            elif isinstance(alleles, list) and trait != 'lineage':
                for i in range(len(alleles)):
                     if random.random() < SimulationSettings.ALLELE_MUTATION_RATE:
                        if trait == 'color': alleles[i] = random.choice(['A', 'B'])
                        elif trait == 'resource_preference': alleles[i] = random.choice(self.env.environment_colors)
        return new_genotype

    def _create_child_properties(self, partner, is_gene_based_sim):
        child_genotype = {}
        if is_gene_based_sim:
            for trait, p1_alleles in self.genotype.items():
                p2_alleles = partner.genotype.get(trait)
                if p2_alleles is None: continue
                if isinstance(p1_alleles, tuple):
                    child_genotype[trait] = (sorted([random.choice(p1_alleles[0]), random.choice(p2_alleles[0])]), sorted([random.choice(p1_alleles[1]), random.choice(p2_alleles[1])]))
                else:
                    child_genotype[trait] = sorted([random.choice(p1_alleles), random.choice(p2_alleles)])
            child_genotype = self._apply_mutation(child_genotype)
        else:
             temp_child = self.__class__(self.env, -1, -1); child_genotype = temp_child._generate_genotype_from_phenotype()
        child_phenotype = self.calculate_phenotype_from_genotype(child_genotype)
        return child_genotype, child_phenotype

    def calculate_genetic_distance(self, other):
        return 1.0 - self._calculate_genetic_similarity(other)

    def _calculate_genetic_similarity(self, other):
        total_sim, total_weight = 0.0, 0.0
        for trait, weight in SimulationSettings.GENETIC_SIMILARITY_WEIGHTS.items():
            if not hasattr(self, trait) or not hasattr(other, trait) or self.genotype.get(trait) is None: continue
            total_weight += weight; gene1, gene2 = self.genotype.get(trait), other.genotype.get(trait); trait_sim = 0.0
            if isinstance(gene1, (list, tuple)):
                alleles1 = (gene1[0] + gene1[1]) if isinstance(gene1[0], list) else gene1
                alleles2 = (gene2[0] + gene2[1]) if isinstance(gene2[0], list) else gene2
                intersection = sum((collections.Counter(alleles1) & collections.Counter(alleles2)).values())
                union = sum((collections.Counter(alleles1) | collections.Counter(alleles2)).values())
                if union > 0: trait_sim = intersection / union
            elif trait == 'learning_rate':
                trait_sim = 1.0 - (abs(float(gene1[0]) - float(gene2[0])) / 0.9)
            total_sim += trait_sim * weight
        return total_sim / total_weight if total_weight > 0 else 1.0
    
    def _calculate_mating_compatibility(self, other):
        score = 1.0
        if self.lineage != other.lineage:
            score *= (1.0 - SimulationSettings.LINEAGE_MISMATCH_PENALTY)
        # Add other compatibility checks here if needed
        return max(0, score)

    def die(self, reason=""):
        if self.is_alive:
            self.is_alive = False
            self.env.remove_individual_from_occupancy_map(self)
            self.env.add_food_from_carcass(self.x, self.y, self.carcass_food_value)
            # self.env.simulation_ref.turn_logs[self.env.current_turn].append(f"ID {self.id} died: {reason}")
    
    def age_and_check_death(self):
        self.age += 1
        actual_max_age = self.max_age * (1 + (self.lifespan_potential - 1) * SimulationSettings.LIFESPAN_POTENTIAL_AGE_BONUS_FACTOR)
        if self.age >= actual_max_age:
            self.die("old age")
            return True
        return False
    
    def get_energy_consumption(self):
        # Placeholder for actual energy calculation
        return 5

    def move(self):
        if not self.is_alive: return
        self.env.remove_individual_from_occupancy_map(self)
        dx, dy = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        new_x, new_y = self.x + dx, self.y + dy
        if self.env.is_valid_position(new_x, new_y) and not self.env.is_barrier(new_x, new_y):
            self.x, self.y = new_x, new_y
        self.env.add_individual_to_occupancy_map(self)

class Prey(Individual):
    def __init__(self, env, x, y, age=0, energy=SimulationSettings.MAX_ENERGY_PREY / 2, parent_id=None, birth_turn=None):
        super().__init__(env, x, y, age, energy, parent_id, birth_turn)
        self.max_energy = SimulationSettings.MAX_ENERGY_PREY
        self.base_energy_consumption = SimulationSettings.BASE_ENERGY_CONSUMPTION_PREY
        self.max_age = SimulationSettings.MAX_AGE_PREY
        self.carcass_food_value = SimulationSettings.CARCASS_FOOD_VALUE_PREY
    
    def eat(self):
        food_eaten = self.env.consume_food(self.x, self.y, 10) # Simplified intake
        self.energy = min(self.max_energy, self.energy + food_eaten)

class Predator(Individual):
    def __init__(self, env, x, y, age=0, energy=SimulationSettings.MAX_ENERGY_PREDATOR / 2, parent_id=None, birth_turn=None):
        super().__init__(env, x, y, age, energy, parent_id, birth_turn)
        self.max_energy = SimulationSettings.MAX_ENERGY_PREDATOR
        self.base_energy_consumption = SimulationSettings.BASE_ENERGY_CONSUMPTION_PREDATOR
        self.max_age = SimulationSettings.MAX_AGE_PREDATOR
        self.carcass_food_value = SimulationSettings.CARCASS_FOOD_VALUE_PREDATOR
        self.hunting_range = SimulationSettings.PREDATOR_HUNTING_RANGE
        self.base_movement_speed = SimulationSettings.BASE_MOVEMENT_SPEED_PREDATOR
    
    def hunt(self, prey_population):
        hunted_prey_id = None
        occupants = self.env.occupancy_map.get((self.x, self.y), set())
        for prey_id in list(occupants):
            prey = prey_population.get(prey_id)
            if prey and prey.is_alive:
                prey.die("hunted")
                self.energy = min(self.max_energy, self.energy + prey.max_energy)
                hunted_prey_id = prey.id
                break # Hunt one prey per turn
        return hunted_prey_id
