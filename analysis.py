# analysis.py
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import collections
from simulation import Simulation
from settings import SimulationSettings

def run_multiple_simulations(num_simulations, sim_type, sim_params, base_seed=None):
    all_sim_results = []
    print(f"\n--- {sim_type} 시뮬레이션 ({num_simulations}회 반복) 시작 ---")
    for i in range(num_simulations):
        current_seed = base_seed + i if base_seed is not None else None
        print(f"  > 시뮬레이션 {i+1}/{num_simulations} 실행 중 (시드: {current_seed})...")
        sim = Simulation(
            width=sim_params['width'], height=sim_params['height'],
            max_food_per_tile=sim_params['max_food_per_tile'], food_regen_rate=sim_params['food_regen_rate'],
            initial_prey_count=sim_params['initial_prey_count'], initial_predator_count=sim_params['initial_predator_count'],
            max_turns=sim_params['max_turns'], is_gene_based_sim=(sim_type == "유전자 기반"),
            seed=current_seed
        )
        while sim.run_turn():
            pass
        all_sim_results.append(sim.get_simulation_results())
        print(f"  > 시뮬레이션 {i+1} 종료 (총 턴: {sim.turn}, 피식자: {len(sim.prey)}, 포식자: {len(sim.predators)})")
    return all_sim_results

def analyze_and_plot_results(sim_results_list, sim_type_label, output_dir, base_seed_used=None):
    if not sim_results_list:
        print("분석할 결과가 없습니다."); return
        
    report_path = os.path.join(output_dir, f"{sim_type_label}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"### {sim_type_label} 시뮬레이션 결과 보고서 ###\n")
        f.write(f"실행 횟수: {len(sim_results_list)}회\n")
        f.write(f"사용된 기본 시드: {base_seed_used if base_seed_used is not None else '랜덤'}\n\n")

    max_turns = max(res['final_turn'] for res in sim_results_list) if sim_results_list else 0

    # Plotting population change
    plt.figure(figsize=(12, 6))
    for i, results in enumerate(sim_results_list):
        turns = sorted(results['data'].keys())
        prey_counts = [results['data'][t].get('prey_count', 0) for t in turns]
        predator_counts = [results['data'][t].get('predator_count', 0) for t in turns]
        plt.plot(turns, prey_counts, alpha=0.5, linestyle='--')
        plt.plot(turns, predator_counts, alpha=0.5)
    plt.title(f'{sim_type_label} - 개별 시뮬레이션별 개체군 크기 변화'); plt.xlabel('턴'); plt.ylabel('개체 수'); plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{sim_type_label}_individual_population.png')); plt.close()

    # Plotting average trait changes
    avg_traits_to_plot = [k for k, v in SimulationSettings.GENETIC_SIMILARITY_WEIGHTS.items()]
    for trait in avg_traits_to_plot:
        plt.figure(figsize=(12, 6))
        for species, color in [('prey', 'green'), ('predator', 'red')]:
            avg_values = np.zeros(max_turns + 1)
            counts = np.zeros(max_turns + 1)
            for results in sim_results_list:
                for t, data in results['data'].items():
                    if t <= max_turns and data.get(f'{species}_count', 0) > 0:
                        trait_data = data.get(f'{species}_traits', {})
                        if f'{trait}_avg' in trait_data:
                            avg_values[t] += trait_data[f'{trait}_avg']
                            counts[t] += 1
            avg_values = np.divide(avg_values, counts, out=np.zeros_like(avg_values), where=counts!=0)
            plt.plot(range(max_turns + 1), avg_values, label=f'평균 {species} {trait}', color=color)
        plt.title(f'{sim_type_label} - 평균 {trait} 변화'); plt.xlabel('턴'); plt.ylabel('평균값'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{sim_type_label}_avg_{trait}.png')); plt.close()

    # Plotting distribution changes (e.g., lineage)
    dist_traits_to_plot = ['lineage', 'color', 'resource_preference']
    for trait in dist_traits_to_plot:
        plt.figure(figsize=(12, 6))
        all_labels = sorted(list(set(l for res in sim_results_list for t_data in res['data'].values() for s_type in ['prey','predator'] for l in t_data.get(f'{s_type}_traits',{}).get(f'{trait}_dist',{}).keys())))
        color_map = {label: plt.get_cmap('gist_rainbow')(i / len(all_labels)) for i, label in enumerate(all_labels)}
        
        for species, line_style in [('prey', '--'), ('predator', '-')]:
            for label in all_labels:
                avg_props = np.zeros(max_turns + 1)
                counts = np.zeros(max_turns + 1)
                for results in sim_results_list:
                    for t, data in results['data'].items():
                        if t <= max_turns and data.get(f'{species}_count', 0) > 0:
                            dist = data.get(f'{species}_traits', {}).get(f'{trait}_dist', {})
                            total = sum(dist.values())
                            if total > 0:
                                avg_props[t] += dist.get(label, 0) / total
                                counts[t] += 1
                avg_props = np.divide(avg_props, counts, out=np.zeros_like(avg_props), where=counts!=0)
                if np.any(avg_props > 0):
                    plt.plot(range(max_turns + 1), avg_props, label=f'{species} {label}', color=color_map[label], linestyle=line_style)
        plt.title(f'{sim_type_label} - 평균 {trait} 분포 변화'); plt.xlabel('턴'); plt.ylabel('비율'); plt.legend(ncol=2); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{sim_type_label}_avg_{trait}_dist.png')); plt.close()

    # Final report summary
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n--- 각 시뮬레이션 요약 ---\n")
        for i, results in enumerate(sim_results_list):
            final_turn_data = results['data'].get(results['final_turn'], {})
            prey_traits = final_turn_data.get('prey_traits', {})
            pred_traits = final_turn_data.get('predator_traits', {})
            
            f.write(f"\n## 시뮬레이션 {i+1}:\n")
            f.write(f"  - 종료 턴: {results['final_turn']}\n")
            f.write(f"  - 최종 피식자/포식자: {sum(prey_traits.get('lineage_dist', {}).values())} / {sum(pred_traits.get('lineage_dist', {}).values())}\n")
            f.write(f"  - 신규 계통 발생 횟수: {results.get('new_lineage_events', 0)}회\n")
            f.write(f"  - 기존 계통 편입 횟수: {results.get('assimilation_events', 0)}회\n")

        f.write("\n--- 모든 시뮬레이션 총 경향성 ---\n")
        avg_new = np.mean([res.get('new_lineage_events', 0) for res in sim_results_list])
        avg_assim = np.mean([res.get('assimilation_events', 0) for res in sim_results_list])
        f.write(f"  - 평균 신규 계통 발생: {avg_new:.2f}회\n")
        f.write(f"  - 평균 기존 계통 편입: {avg_assim:.2f}회\n")

    print(f"\n분석 완료. 결과는 '{output_dir}' 폴더에 저장되었습니다.")
