import os
import sys
import subprocess
import random
import numpy as np
import traci
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# --- SUMO Configuration ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

SUMO_BINARY = "sumo"
CONFIG_FILE = r"D:\Class notes and stuff\Sem 5\Soft_Computing\CN+ Dataset\CN+ Dataset\SUMO Format\Infrastructure Files\CN+ Dataset.sumocfg"
TRIPINFO_FILE_BASELINE = "tripinfo_baseline.xml"
TRIPINFO_FILE_GA = "tripinfo_ga_run.xml"

# --- GA and ANN Configuration ---
JUNCTION_ID = '25350584'
INCOMING_LANES = ['189004526#1_0', '31014214_0', '31014214_1']
GREEN_PHASES = [0, 4]
YELLOW_PHASES = {0: 1, 4: 5}
YELLOW_PHASE_DURATION = 4
INPUT_NODES = len(INCOMING_LANES)
HIDDEN_NODES = 10
OUTPUT_NODES = 1
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
ELITISM_COUNT = 1
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60

# --- ANN Class (No changes needed) ---
class ANN:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes, self.hidden_nodes, self.output_nodes = input_nodes, hidden_nodes, output_nodes
        self.weights_ih = np.random.randn(self.hidden_nodes, self.input_nodes)
        self.weights_ho = np.random.randn(self.output_nodes, self.hidden_nodes)
        self.bias_h = np.random.randn(self.hidden_nodes, 1)
        self.bias_o = np.random.randn(self.output_nodes, 1)
    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def predict(self, inputs):
        inputs_vec = np.array(inputs, ndmin=2).T
        hidden = self._sigmoid(np.dot(self.weights_ih, inputs_vec) + self.bias_h)
        output = self._sigmoid(np.dot(self.weights_ho, hidden) + self.bias_o)
        return MIN_GREEN_TIME + (output.item() * (MAX_GREEN_TIME - MIN_GREEN_TIME))
    def get_flat_weights(self):
        return np.concatenate([w.flatten() for w in [self.weights_ih, self.weights_ho, self.bias_h, self.bias_o]])
    def set_weights_from_flat(self, flat_weights):
        idx=0; w_ih_size=self.hidden_nodes*self.input_nodes; self.weights_ih=flat_weights[idx:idx+w_ih_size].reshape(self.hidden_nodes, self.input_nodes)
        idx+=w_ih_size; w_ho_size=self.output_nodes*self.hidden_nodes; self.weights_ho=flat_weights[idx:idx+w_ho_size].reshape(self.output_nodes, self.hidden_nodes)
        idx+=w_ho_size; b_h_size=self.hidden_nodes; self.bias_h=flat_weights[idx:idx+b_h_size].reshape(self.hidden_nodes, 1)
        idx+=b_h_size; b_o_size=self.output_nodes; self.bias_o=flat_weights[idx:idx+b_o_size].reshape(self.output_nodes, 1)

# --- Simulation and Fitness Evaluation ---
def parse_tripinfo_waiting_time(tripinfo_file):
    """Parses only the total waiting time for fast fitness evaluation."""
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        if not root.findall('tripinfo'): return float('inf')
        return sum(float(trip.get('waitingTime')) for trip in root.findall('tripinfo'))
    except (ET.ParseError, FileNotFoundError):
        return float('inf')

# *** NEW: Detailed parsing function for final report graphs ***
def parse_detailed_tripinfo(tripinfo_file):
    """Parses detailed metrics from a tripinfo file for reporting."""
    wait_times = []
    durations = []
    time_losses = []
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        trips = root.findall('tripinfo')
        if not trips:
            return {'total_wait': float('inf'), 'total_duration': float('inf'), 
                    'total_time_loss': float('inf'), 'wait_times_list': []}
        
        for trip in trips:
            wait_times.append(float(trip.get('waitingTime')))
            durations.append(float(trip.get('duration')))
            time_losses.append(float(trip.get('timeLoss')))
            
        return {
            'total_wait': sum(wait_times),
            'total_duration': sum(durations),
            'total_time_loss': sum(time_losses),
            'wait_times_list': wait_times
        }
    except (ET.ParseError, FileNotFoundError):
        return {'total_wait': float('inf'), 'total_duration': float('inf'), 
                'total_time_loss': float('inf'), 'wait_times_list': []}


def run_simulation(chromosome=None, is_baseline_run=False):
    """Runs a single SUMO simulation."""
    output_file = TRIPINFO_FILE_BASELINE if is_baseline_run else TRIPINFO_FILE_GA
    
    if chromosome is not None:
        # GA-ANN controlled run
        sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--tripinfo-output", output_file, "--random"]
        traci.start(sumo_cmd)
        controller = ANN(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
        controller.set_weights_from_flat(chromosome)
        next_green_phase_idx = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            current_phase = traci.trafficlight.getPhase(JUNCTION_ID)
            if current_phase in GREEN_PHASES:
                inputs = [traci.lane.getLastStepHaltingNumber(lane) for lane in INCOMING_LANES]
                green_duration = controller.predict(inputs)
                start_time = traci.simulation.getTime()
                while traci.simulation.getTime() < start_time + green_duration:
                    if traci.simulation.getMinExpectedNumber() == 0: break
                    traci.simulationStep()
                if traci.simulation.getMinExpectedNumber() == 0: break
                
                yellow_phase = YELLOW_PHASES[current_phase]
                traci.trafficlight.setPhase(JUNCTION_ID, yellow_phase)
                start_time = traci.simulation.getTime()
                while traci.simulation.getTime() < start_time + YELLOW_PHASE_DURATION:
                    if traci.simulation.getMinExpectedNumber() == 0: break
                    traci.simulationStep()
                if traci.simulation.getMinExpectedNumber() == 0: break
                
                next_green_phase_idx = 1 - next_green_phase_idx
                traci.trafficlight.setPhase(JUNCTION_ID, GREEN_PHASES[next_green_phase_idx])
            else:
                traci.simulationStep()
        traci.close()
    else: 
        # Baseline (fixed-time) run
        sumo_cmd = [SUMO_BINARY, "-c", CONFIG_FILE, "--tripinfo-output", output_file]
        subprocess.run(sumo_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Return only total waiting time for fitness calculation during training
    # For detailed analysis, we'll call the detailed parser separately
    if not is_baseline_run:
        return parse_tripinfo_waiting_time(output_file)
    else:
        # For the initial baseline run, we also just need the total wait time
        return parse_tripinfo_waiting_time(output_file)


# --- Genetic Algorithm Functions (No changes needed) ---
def create_individual():
    return ANN(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES).get_flat_weights()

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 2)
        return np.concatenate([parent1[:point], parent2[point:]]), np.concatenate([parent2[:point], parent1[point:]])
    return parent1, parent2

def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] += np.random.normal(0, 0.2)
    return chromosome

def selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), k=3)
        tournament.sort(key=lambda x: x[1], reverse=True)
        selected.append(tournament[0][0])
    return selected

# --- Main Execution ---
if __name__ == "__main__":
    print("--- SUMO Traffic Control with GA-ANN (Training & Evaluation) ---")
    
    # --- [1/6] Baseline Simulation ---
    print("\n[1/6] Running baseline simulation...")
    baseline_wait_time = run_simulation(is_baseline_run=True)
    if baseline_wait_time == float('inf'): sys.exit("Error: Baseline simulation failed.")
    print(f"-> Baseline Total Waiting Time: {baseline_wait_time:.2f} seconds")

    # --- [2/6] Genetic Algorithm Training ---
    print("\n[2/6] Starting Genetic Algorithm training...")
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_overall_chromosome = None
    best_overall_fitness = -1
    
    # *** MODIFIED: Store history for best, average, and worst for plotting ***
    best_wait_times_history = []
    avg_wait_times_history = []
    worst_wait_times_history = []

    for gen in range(NUM_GENERATIONS):
        print(f"\n--- Generation {gen + 1}/{NUM_GENERATIONS} ---")
        
        # Evaluate all individuals in the population
        fitnesses = [1.0 / (1.0 + run_simulation(chromosome=ind)) for ind in population]
        
        # *** NEW: Calculate stats for this generation ***
        wait_times_gen = [(1.0 / f) - 1.0 for f in fitnesses]
        best_gen_wait_time = min(wait_times_gen)
        avg_gen_wait_time = np.mean(wait_times_gen)
        worst_gen_wait_time = max(wait_times_gen)
        
        best_wait_times_history.append(best_gen_wait_time)
        avg_wait_times_history.append(avg_gen_wait_time)
        worst_wait_times_history.append(worst_gen_wait_time)
        
        best_gen_fitness = max(fitnesses)
        best_gen_idx = fitnesses.index(best_gen_fitness)
        
        print(f"  Best Wait Time: {best_gen_wait_time:.2f}s")
        print(f"  Avg Wait Time:  {avg_gen_wait_time:.2f}s")
        print(f"  Worst Wait Time: {worst_gen_wait_time:.2f}s")

        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_chromosome = population[best_gen_idx].copy()
            print(f"  *** New overall best found! ***")
        
        # Create new population
        new_population = []
        sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda p: p[0], reverse=True)]
        new_population.extend(sorted_pop[:ELITISM_COUNT])
        selected_parents = selection(population, fitnesses)
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = random.sample(selected_parents, k=2)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < POPULATION_SIZE: new_population.append(mutate(c2))
        population = new_population

    print("\n[3/6] GA training complete.")

    # --- [4/6] Save the Best Model ---
    print("\n[4/6] Saving the best model...")
    model_filename = "best_ann_weights.npy"
    np.save(model_filename, best_overall_chromosome)
    print(f"-> Best model weights saved to {model_filename}")

    # --- [5/6] Re-run for Detailed Analysis ---
    # We re-run the baseline and the best GA controller to get clean,
    # detailed XML files for our final report graphs.
    print("\n[5/6] Re-running simulations for detailed analysis...")
    
    print("-> Running final baseline simulation...")
    run_simulation(is_baseline_run=True) # This creates TRIPINFO_FILE_BASELINE
    baseline_metrics = parse_detailed_tripinfo(TRIPINFO_FILE_BASELINE)
    print(f"-> Baseline Total Wait: {baseline_metrics['total_wait']:.2f}s")

    print("-> Running final optimized simulation...")
    run_simulation(chromosome=best_overall_chromosome, is_baseline_run=False) # This creates TRIPINFO_FILE_GA
    ga_metrics = parse_detailed_tripinfo(TRIPINFO_FILE_GA)
    print(f"-> Optimized Total Wait: {ga_metrics['total_wait']:.2f}s")

    print("\n--- Final Results Summary ---")
    print(f"Baseline Waiting Time: {baseline_metrics['total_wait']:.2f} seconds")
    print(f"Optimized Waiting Time: {ga_metrics['total_wait']:.2f} seconds")
    if baseline_metrics['total_wait'] > 0:
        improvement = ((baseline_metrics['total_wait'] - ga_metrics['total_wait']) / baseline_metrics['total_wait']) * 100
        print(f"\nImprovement in Waiting Time: {improvement:.2f}%")


    # --- [6/6] PLOTTING SECTION ---
    print("\n[6/6] Generating performance plots...")

    # --- Plot 1: GA Convergence (Best, Avg, Worst) ---
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, NUM_GENERATIONS + 1), best_wait_times_history, marker='o', linestyle='-', label='Best Individual')
    plt.plot(range(1, NUM_GENERATIONS + 1), avg_wait_times_history, marker='x', linestyle='--', label='Generation Average')
    plt.plot(range(1, NUM_GENERATIONS + 1), worst_wait_times_history, marker='s', linestyle=':', label='Worst Individual')
    plt.axhline(y=baseline_wait_time, color='r', linestyle='--', label=f'Baseline ({baseline_wait_time:.2f}s)')
    plt.title('GA Convergence Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Total Waiting Time (s)')
    plt.xticks(range(1, NUM_GENERATIONS + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ga_convergence_plot.png")
    print("-> Plot 1 saved to ga_convergence_plot.png")
    plt.close()

    # --- Plot 2: Final Performance Bar Chart (Wait Time) ---
    plt.figure(figsize=(8, 6))
    controllers = ['Baseline (Fixed)', 'GA-ANN (Optimized)']
    wait_times = [baseline_metrics['total_wait'], ga_metrics['total_wait']]
    colors = ['#d9534f', '#5cb85c']
    bars = plt.bar(controllers, wait_times, color=colors)
    plt.ylabel('Total Waiting Time (s)')
    plt.title('Controller Performance Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(wait_times)*0.01, f'{yval:.2f}s', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("summary_bar_chart.png")
    print("-> Plot 2 saved to summary_bar_chart.png")
    plt.close()

    # --- Plot 3: Detailed Metrics Comparison (Wait, Duration, TimeLoss) ---
    labels = ['Total Waiting Time', 'Total Travel Time', 'Total Time Loss']
    baseline_values = [baseline_metrics['total_wait'], baseline_metrics['total_duration'], baseline_metrics['total_time_loss']]
    ga_values = [ga_metrics['total_wait'], ga_metrics['total_duration'], ga_metrics['total_time_loss']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#d9534f')
    rects2 = ax.bar(x + width/2, ga_values, width, label='GA-ANN', color='#5cb85c')
    
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Detailed Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.bar_label(rects1, padding=3, fmt='%.0f')
    ax.bar_label(rects2, padding=3, fmt='%.0f')
    
    fig.tight_layout()
    plt.savefig("detailed_metrics_comparison.png")
    print("-> Plot 3 saved to detailed_metrics_comparison.png")
    plt.close()

    # --- Plot 4: Histogram of Individual Waiting Times ---
    plt.figure(figsize=(12, 7))
    max_wait = max(max(baseline_metrics['wait_times_list']), max(ga_metrics['wait_times_list']))
    bins = np.linspace(0, max_wait, 50)
    plt.hist(baseline_metrics['wait_times_list'], bins=bins, alpha=0.7, label='Baseline', color='#d9534f')
    plt.hist(ga_metrics['wait_times_list'], bins=bins, alpha=0.7, label='GA-ANN', color='#5cb85c')
    plt.title('Distribution of Individual Vehicle Waiting Times')
    plt.xlabel('Waiting Time per Vehicle (s)')
    plt.ylabel('Number of Vehicles')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("wait_time_histogram.png")
    print("-> Plot 4 saved to wait_time_histogram.png")
    plt.close()

    # --- Plot 5: Box Plot of Individual Waiting Times ---
    plt.figure(figsize=(8, 6))
    data_to_plot = [baseline_metrics['wait_times_list'], ga_metrics['wait_times_list']]
    box = plt.boxplot(data_to_plot, patch_artist=True, labels=['Baseline', 'GA-ANN'], 
                      showfliers=False) # showfliers=False to avoid extreme outliers skewing the view
    
    colors = ['#d9534f', '#5cb85c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    plt.title('Box Plot of Individual Vehicle Waiting Times')
    plt.ylabel('Waiting Time (s)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("wait_time_boxplot.png")
    print("-> Plot 5 saved to wait_time_boxplot.png")
    plt.close()


    # --- [7/7] Final Cleanup ---
    print("\n[7/7] Cleaning up temporary files...")
    if os.path.exists(TRIPINFO_FILE_BASELINE): os.remove(TRIPINFO_FILE_BASELINE)
    if os.path.exists(TRIPINFO_FILE_GA): os.remove(TRIPINFO_FILE_GA)
    print("-> Done.")