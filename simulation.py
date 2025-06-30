# simulation.py
import random
import collections
from settings import SimulationSettings
from environment import Environment
from individuals import Prey, Predator, Individual

class Simulation:
    """전체 시뮬레이션을 제어하고 실행하는 클래스입니다."""
    def __init__(self, width, height, max_food_per_tile, food_regen_rate,
                         initial_prey_count, initial_predator_count,
                         max_turns, is_gene_based_sim, seed=None ):
        if seed is not None:
            random.seed(seed)
        
        self.env = Environment(width, height, max_food_per_tile, food_regen_rate)
        self.env.simulation_ref = self
        self.prey, self.predators = {}, {}
        self.max_turns = max_turns
        self.is_gene_based_sim = is_gene_based_sim
        self.turn = 0
        self.simulation_data = collections.defaultdict(dict)
        self.turn_logs = collections.defaultdict(list)
        self.new_lineage_events, self.assimilation_events = 0, 0
        
        self.spatial_hash_grid_size = SimulationSettings.SPATIAL_HASH_GRID_SIZE
        self.spatial_hash_map = collections.defaultdict(list)

        self._initialize_population(initial_prey_count, initial_predator_count)

    def _update_spatial_hash(self):
        """매 턴 모든 개체의 위치를 기반으로 공간 해시맵을 새로 고칩니다."""
        self.spatial_hash_map.clear()
        all_individuals = list(self.prey.values()) + list(self.predators.values())
        for ind in all_individuals:
            if ind.is_alive:
                grid_x = ind.x // self.spatial_hash_grid_size
                grid_y = ind.y // self.spatial_hash_grid_size
                self.spatial_hash_map[(grid_x, grid_y)].append(ind.id)
    
    def _get_nearby_ids(self, individual):
        """공간 해시맵을 사용해 개체 주변의 후보 ID 목록을 반환합니다."""
        nearby_ids = []
        grid_x = individual.x // self.spatial_hash_grid_size
        grid_y = individual.y // self.spatial_hash_grid_size
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbor_grid_key = (grid_x + i, grid_y + j)
                candidate_ids = self.spatial_hash_map.get(neighbor_grid_key, [])
                nearby_ids.extend(candidate_ids)
        return nearby_ids
        
    def run_turn(self):
        self.turn += 1; self.env.current_turn = self.turn
        self.env.update_food(); self.env.update_barriers(self.turn)
        
        # [수정] 이동 후 공간 해시맵 업데이트
        all_individuals = list(self.prey.values()) + list(self.predators.values()); random.shuffle(all_individuals)
        for ind in all_individuals:
            if ind.is_alive: ind.move()
        self._update_spatial_hash()

        new_offsprings, dead_ids = [], set()

        for ind in all_individuals:
            if not ind.is_alive: continue
            
            # 1. 생명 활동 (노화, 에너지 소비)
            ind.energy -= ind.get_energy_consumption()
            if ind.energy <= 0: ind.die("starvation")
            if ind.age_and_check_death(): dead_ids.add(ind.id); continue

            # 2. 종별 행동 (사냥, 식사) 및 번식 (공간 해싱 활용)
            population = self.prey if isinstance(ind, Prey) else self.predators
            nearby_ids = self._get_nearby_ids(ind)

            if isinstance(ind, Predator):
                hunted_id = ind.hunt(self.prey, nearby_ids) # 주변 개체만 대상으로 사냥
                if hunted_id: dead_ids.add(hunted_id)
            elif isinstance(ind, Prey):
                ind.eat()

            # 번식
            min_repro_energy = ind.max_energy * (SimulationSettings.PREY_MIN_REPRO_ENERGY_FACTOR if isinstance(ind, Prey) else SimulationSettings.PREDATOR_MIN_REPRO_ENERGY_FACTOR)
            if ind.age >= ind.reproduction_cycle and ind.energy >= min_repro_energy:
                for other_id in nearby_ids: # 전체가 아닌 주변 개체만 확인
                    if other_id == ind.id: continue
                    partner = population.get(other_id)
                    if partner and partner.is_alive and partner.energy >= min_repro_energy and (partner.x, partner.y) == (ind.x, ind.y):  
                        compatibility = ind._calculate_mating_compatibility(partner)
                        
                        if random.random() < compatibility:
                            child_genotype, child_phenotype = ind._create_child_properties(partner, self.is_gene_based_sim)
                            child_class = ind.__class__
                            new_child = child_class(self.env, ind.x, ind.y, parent_id=ind.id)
                            new_child.genotype = child_genotype
                            for trait, value in child_phenotype.items():
                                setattr(new_child, trait, value)
                            new_offsprings.append(new_child)
                            
                            ind.energy -= ind.max_energy * (SimulationSettings.PREY_REPRO_ENERGY_COST_FACTOR if isinstance(ind, Prey) else SimulationSettings.PREDATOR_REPRO_ENERGY_COST_FACTOR)
                            partner.energy -= partner.max_energy * (SimulationSettings.PREY_REPRO_ENERGY_COST_FACTOR if isinstance(partner, Prey) else SimulationSettings.PREDATOR_REPRO_ENERGY_COST_FACTOR)
                            break # 한 턴에 한 파트너와만 번식

        self._process_offspring(new_offsprings)
        self._remove_dead(dead_ids)
        self._collect_data()
        
        if not self.prey or not self.predators or self.turn >= self.max_turns:
            return False
        return True

    def _process_offspring(self, offspring_list):
        for child in offspring_list:
            parent_pop = self.prey if isinstance(child, Prey) else self.predators
            parent = parent_pop.get(child.parent_id)
            
            if not parent:
                parent_pop[child.id] = child
                continue
            
            distance_to_parent = child.calculate_genetic_distance(parent)

            if distance_to_parent <= SimulationSettings.SPECIATION_THRESHOLD:
                child.lineage = parent.lineage
                child.genotype['lineage'] = list(parent.genotype['lineage'])
            else:
                other_lineages = {p.lineage for p in parent_pop.values() if p.lineage != parent.lineage}
                closest_lineage, min_dist = None, float('inf')
                
                for lineage_name in other_lineages:
                    representatives = [p for p in parent_pop.values() if p.lineage == lineage_name]
                    if not representatives: continue
                    representative = random.choice(representatives)
                    distance = child.calculate_genetic_distance(representative)
                    if distance < min_dist:
                        min_dist, closest_lineage = distance, lineage_name
                
                if closest_lineage and min_dist < SimulationSettings.SPECIATION_THRESHOLD:
                    self.assimilation_events += 1
                    child.lineage = closest_lineage
                    child.genotype['lineage'] = [closest_lineage] * 2
                    self.turn_logs[self.turn].append(f"*** 계통 편입! *** {type(child).__name__} {child.id} -> '{closest_lineage}'")
                else:
                    self.new_lineage_events += 1
                    all_names = {p.lineage for p in self.prey.values()} | {p.lineage for p in self.predators.values()}
                    last_char = sorted(list(all_names))[-1] if all_names else '@'
                    new_name = chr(ord(last_char) + 1)
                    if len(new_name) > 1 or not new_name.isalpha(): # Handle non-alphabetic/multi-char results
                         new_name = 'A' if not all_names else last_char + 'A'
                    
                    child.lineage = new_name
                    child.genotype['lineage'] = [new_name] * 2
                    self.turn_logs[self.turn].append(f"*** 신규 계통 발생! *** {type(child).__name__} {child.id} -> '{new_name}'")
            
            parent_pop[child.id] = child
    
    def _remove_dead(self, dead_ids):
        for dead_id in dead_ids:
            if dead_id in self.prey:
                if self.prey[dead_id].is_alive: self.prey[dead_id].die("removed")
                del self.prey[dead_id]
            if dead_id in self.predators:
                if self.predators[dead_id].is_alive: self.predators[dead_id].die("removed")
                del self.predators[dead_id]

    def get_simulation_results(self):
        return {
            'data': self.simulation_data, 'final_turn': self.turn,
            'prey_alive': len(self.prey) > 0, 'predator_alive': len(self.predators) > 0,
            'turn_logs': self.turn_logs, 'new_lineage_events': self.new_lineage_events,
            'assimilation_events': self.assimilation_events
        }

    def _collect_data(self):
        self.simulation_data[self.turn]['prey_count'] = len(self.prey)
        self.simulation_data[self.turn]['predator_count'] = len(self.predators)
        
        prey_traits = collections.defaultdict(list)
        for p in self.prey.values():
            for trait in SimulationSettings.GENETIC_SIMILARITY_WEIGHTS.keys():
                if hasattr(p, trait): prey_traits[trait].append(getattr(p, trait))
            prey_traits['lineage'].append(p.lineage)
        
        pred_traits = collections.defaultdict(list)
        for p in self.predators.values():
            for trait in SimulationSettings.GENETIC_SIMILARITY_WEIGHTS.keys():
                 if hasattr(p, trait): pred_traits[trait].append(getattr(p, trait))
            pred_traits['lineage'].append(p.lineage)

        def get_trait_summary(traits_dict):
            summary = {}
            if not traits_dict: return summary
            for k, v in traits_dict.items():
                if not v: continue
                if isinstance(v[0], (int, float)):
                    summary[f'{k}_avg'] = np.mean(v)
                elif isinstance(v[0], str):
                    summary[f'{k}_dist'] = collections.Counter(v)
            return summary

        self.simulation_data[self.turn]['prey_traits'] = get_trait_summary(prey_traits)
        self.simulation_data[self.turn]['predator_traits'] = get_trait_summary(pred_traits)
