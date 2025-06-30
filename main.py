# main.py
import os
import datetime
from settings import SimulationSettings
from analysis import run_multiple_simulations, analyze_and_plot_results

if __name__ == "__main__":
    # 시뮬레이션 공통 파라미터 설정
    sim_common_params = {
        'width': SimulationSettings.WIDTH,
        'height': SimulationSettings.HEIGHT,
        'max_food_per_tile': SimulationSettings.MAX_FOOD_PER_TILE,
        'food_regen_rate': SimulationSettings.FOOD_REGEN_RATE,
        'initial_prey_count': SimulationSettings.INITIAL_PREY_COUNT,
        'initial_predator_count': SimulationSettings.INITIAL_PREDATOR_COUNT,
        'max_turns': SimulationSettings.MAX_TURNS
    }

    num_sims_to_run = 5 # 각 시나리오별로 실행할 시뮬레이션 횟수

    # 결과 보고서와 그래프를 저장할 메인 폴더 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base_dir = f"simulation_results_{timestamp}"
    os.makedirs(results_base_dir, exist_ok=True)

    # 유전자 기반 시뮬레이션 실행 및 결과 분석
    gene_based_output_dir = os.path.join(results_base_dir, "gene_based")
    os.makedirs(gene_based_output_dir, exist_ok=True)
    gene_based_results = run_multiple_simulations(
        num_simulations=num_sims_to_run, 
        sim_type="유전자 기반", 
        sim_params=sim_common_params, 
        base_seed=123 # 재현성을 위해 시드 고정
    )
    analyze_and_plot_results(
        sim_results_list=gene_based_results, 
        sim_type_label="유전자 기반", 
        output_dir=gene_based_output_dir, 
        base_seed_used=123
    )

    # 무작위 시뮬레이션 실행 및 결과 분석
    random_output_dir = os.path.join(results_base_dir, "random")
    os.makedirs(random_output_dir, exist_ok=True)
    random_results = run_multiple_simulations(
        num_simulations=num_sims_to_run, 
        sim_type="무작위", 
        sim_params=sim_common_params, 
        base_seed=456 # 재현성을 위해 시드 고정
    )
    analyze_and_plot_results(
        sim_results_list=random_results, 
        sim_type_label="무작위", 
        output_dir=random_output_dir, 
        base_seed_used=456
    )

    print("\n모든 시뮬레이션 및 분석이 완료되었습니다.")
