import os
import sys
import numpy as np
import traci
import xml.etree.ElementTree as ET
import subprocess
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 20,      
    'axes.titlesize': 26,    
    'axes.labelsize': 22,    
    'xtick.labelsize': 18,    
    'ytick.labelsize': 18,  
    'legend.fontsize': 20,    
    'figure.titlesize': 30   
})
# --- Configuration ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# *** IMPORTANT: Change this to your "next day" or test traffic file ***
TEST_CONFIG_FILE = r"C:\Users\kppse\OneDrive\Desktop\CN+ Dataset\CN+ Dataset\SUMO Format\Infrastructure Files\CN+ Dataset.sumocfg"

SAVED_MODEL_FILE = "best_ann_weights.npy"
# Use "sumo-gui" to watch the test run, "sumo" for faster execution
SUMO_BINARY_TEST = "sumo-gui" 
SUMO_BINARY_BASELINE = "sumo-gui"

# --- ANN and Simulation Constants (copy from training script) ---
JUNCTION_ID = '25350584'
INCOMING_LANES = ['189004526#1_0', '31014214_0', '31014214_1']
GREEN_PHASES = [0, 4]
YELLOW_PHASES = {0: 1, 4: 5}
YELLOW_PHASE_DURATION = 4
INPUT_NODES = len(INCOMING_LANES)
HIDDEN_NODES = 10
OUTPUT_NODES = 1
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60

# --- ANN Class (copy from training script) ---
class ANN:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes, self.hidden_nodes, self.output_nodes = input_nodes, hidden_nodes, output_nodes
        self.weights_ih = np.zeros((self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.zeros((self.output_nodes, self.hidden_nodes))
        self.bias_h = np.zeros((self.hidden_nodes, 1))
        self.bias_o = np.zeros((self.output_nodes, 1))
    def _sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def predict(self, inputs):
        inputs_vec = np.array(inputs, ndmin=2).T
        hidden = self._sigmoid(np.dot(self.weights_ih, inputs_vec) + self.bias_h)
        output = self._sigmoid(np.dot(self.weights_ho, hidden) + self.bias_o)
        return MIN_GREEN_TIME + (output.item() * (MAX_GREEN_TIME - MIN_GREEN_TIME))
    def set_weights_from_flat(self, flat_weights):
        idx=0; w_ih_size=self.hidden_nodes*self.input_nodes; self.weights_ih=flat_weights[idx:idx+w_ih_size].reshape(self.hidden_nodes, self.input_nodes)
        idx+=w_ih_size; w_ho_size=self.output_nodes*self.hidden_nodes; self.weights_ho=flat_weights[idx:idx+w_ho_size].reshape(self.output_nodes, self.hidden_nodes)
        idx+=w_ho_size; b_h_size=self.hidden_nodes; self.bias_h=flat_weights[idx:idx+b_h_size].reshape(self.hidden_nodes, 1)
        idx+=b_h_size; b_o_size=self.output_nodes; self.bias_o=flat_weights[idx:idx+b_o_size].reshape(self.output_nodes, 1)

# --- NEW: Enhanced Data Parsing and Plotting Functions ---

# *** MODIFIED: Renamed and enhanced to get all data lists ***
def parse_detailed_tripinfo(tripinfo_file):
    """Parses a tripinfo XML and returns a dictionary of detailed performance metrics."""
    wait_times = []
    travel_times = []
    time_losses = []
    
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        trips = root.findall('tripinfo')
        if not trips:
            return {
                'total_wait_time': float('inf'), 'avg_travel_time': float('inf'),
                'total_time_loss': float('inf'), 'throughput': 0,
                'wait_times_list': [], 'travel_times_list': [], 'time_losses_list': []
            }

        for trip in trips:
            wait_times.append(float(trip.get('waitingTime')))
            travel_times.append(float(trip.get('duration')))
            time_losses.append(float(trip.get('timeLoss')))
            
        throughput = len(trips)
        total_wait_time = sum(wait_times)
        total_travel_time = sum(travel_times)
        total_time_loss = sum(time_losses)
        avg_travel_time = total_travel_time / throughput if throughput > 0 else 0
        
        return {
            'total_wait_time': total_wait_time,
            'total_travel_time': total_travel_time,
            'total_time_loss': total_time_loss,
            'avg_travel_time': avg_travel_time,
            'throughput': throughput,
            'wait_times_list': wait_times,
            'travel_times_list': travel_times,
            'time_losses_list': time_losses
        }
    except (ET.ParseError, FileNotFoundError):
        # Return an empty/infinite structure on failure
        return {
            'total_wait_time': float('inf'), 'avg_travel_time': float('inf'),
            'total_time_loss': float('inf'), 'throughput': 0,
            'wait_times_list': [], 'travel_times_list': [], 'time_losses_list': []
        }

# *** MODIFIED: Updated to include Time Loss ***
def plot_performance_comparison(baseline_stats, model_stats):
    """Generates a bar chart comparing baseline and model performance."""
    labels = ['Total Waiting Time (s)', 'Total Time Loss (s)', 'Avg. Travel Time (s)', 'Vehicle Throughput']
    baseline_values = [
        baseline_stats['total_wait_time'], 
        baseline_stats['total_time_loss'],
        baseline_stats['avg_travel_time'], 
        baseline_stats['throughput']
    ]
    model_values = [
        model_stats['total_wait_time'], 
        model_stats['total_time_loss'],
        model_stats['avg_travel_time'], 
        model_stats['throughput']
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#d9534f')
    rects2 = ax.bar(x + width/2, model_values, width, label='GA-ANN Model', color='#5cb85c')

    ax.set_ylabel('Values')
    ax.set_title('Performance Comparison: Baseline vs. GA-ANN Model (Test Data)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig("test_performance_comparison.png")
    print("-> Plot 1/5 saved to 'test_performance_comparison.png'")
    plt.close(fig)

def plot_queue_length(baseline_data, model_data):
    """Generates a line chart comparing queue lengths over time."""
    plt.figure(figsize=(12, 6))
    
    if baseline_data:
        time_base, queue_base = zip(*baseline_data)
        plt.plot(time_base, queue_base, label='Baseline Controller', color='#d9534f', alpha=0.8)
    
    if model_data:
        time_model, queue_model = zip(*model_data)
        plt.plot(time_model, queue_model, label='GA-ANN Controller', color='#5cb85c', linewidth=2)
    
    plt.title('Queue Length Over Time (Test Data)')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Total Halting Vehicles (All Monitored Lanes)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_queue_length.png")
    print("-> Plot 2/5 saved to 'test_queue_length.png'")
    plt.close()

# *** NEW: Plot for Waiting Time Distribution (Histogram) ***
def plot_wait_time_histogram(baseline_stats, model_stats):
    """Generates a histogram comparing individual vehicle waiting times."""
    plt.figure(figsize=(12, 7))
    
    # Determine common bins
    max_wait = max(max(baseline_stats['wait_times_list'], default=0), 
                   max(model_stats['wait_times_list'], default=0))
    bins = np.linspace(0, max_wait + 1, 50)
    
    plt.hist(baseline_stats['wait_times_list'], bins=bins, alpha=0.7, label='Baseline', color='#d9534f')
    plt.hist(model_stats['wait_times_list'], bins=bins, alpha=0.7, label='GA-ANN', color='#5cb85c')
    
    plt.title('Distribution of Individual Vehicle Waiting Times (Test Data)')
    plt.xlabel('Waiting Time per Vehicle (s)')
    plt.ylabel('Number of Vehicles')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("test_wait_time_histogram.png")
    print("-> Plot 3/5 saved to 'test_wait_time_histogram.png'")
    plt.close()

# *** NEW: Plot for Waiting Time Distribution (Box Plot) ***
def plot_wait_time_boxplot(baseline_stats, model_stats):
    """Generates a box plot comparing individual vehicle waiting times."""
    plt.figure(figsize=(8, 6))
    
    data_to_plot = [baseline_stats['wait_times_list'], model_stats['wait_times_list']]
    
    box = plt.boxplot(data_to_plot, patch_artist=True, labels=['Baseline', 'GA-ANN'], 
                      showfliers=False) # Hiding outliers for a clearer view of the main distribution
    
    colors = ['#d9534f', '#5cb85c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    plt.title('Box Plot of Individual Vehicle Waiting Times (Test Data)')
    plt.ylabel('Waiting Time (s)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("test_wait_time_boxplot.png")
    print("-> Plot 4/5 saved to 'test_wait_time_boxplot.png'")
    plt.close()

# *** NEW: Plot for Travel Time and Time Loss Distributions ***
def plot_distribution_histograms(baseline_stats, model_stats):
    """Generates histograms for travel time and time loss distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Travel Time Plot ---
    max_travel = max(max(baseline_stats['travel_times_list'], default=0), 
                       max(model_stats['travel_times_list'], default=0))
    bins_travel = np.linspace(0, max_travel + 1, 50)
    
    ax1.hist(baseline_stats['travel_times_list'], bins=bins_travel, alpha=0.7, label='Baseline', color='#d9534f')
    ax1.hist(model_stats['travel_times_list'], bins=bins_travel, alpha=0.7, label='GA-ANN', color='#5cb85c')
    ax1.set_title('Distribution of Individual Vehicle Travel Times')
    ax1.set_xlabel('Travel Time per Vehicle (s)')
    ax1.set_ylabel('Number of Vehicles')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Time Loss Plot ---
    max_loss = max(max(baseline_stats['time_losses_list'], default=0), 
                   max(model_stats['time_losses_list'], default=0))
    bins_loss = np.linspace(0, max_loss + 1, 50)
    
    ax2.hist(baseline_stats['time_losses_list'], bins=bins_loss, alpha=0.7, label='Baseline', color='#d9534f')
    ax2.hist(model_stats['time_losses_list'], bins=bins_loss, alpha=0.7, label='GA-ANN', color='#5cb85c')
    ax2.set_title('Distribution of Individual Vehicle Time Loss')
    ax2.set_xlabel('Time Loss per Vehicle (s)')
    ax2.set_ylabel('Number of Vehicles')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle('Vehicle Time Distributions (Test Data)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("test_time_distributions.png")
    print("-> Plot 5/5 saved to 'test_time_distributions.png'")
    plt.close(fig)


# --- MODIFIED: Simulation Functions now collect more data ---

def run_baseline_simulation(config_file):
    """Runs a baseline simulation and collects queue and trip data."""
    tripinfo_file = "tripinfo_test_baseline.xml"
    sumo_cmd = [SUMO_BINARY_BASELINE, "-c", config_file, "--tripinfo-output", tripinfo_file]
    
    traci.start(sumo_cmd)
    queue_data = []
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_time = traci.simulation.getTime()
        # Get total halting cars on all monitored lanes
        halting_cars = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in INCOMING_LANES)
        queue_data.append((current_time, halting_cars))
    traci.close()
    
    # *** MODIFIED: Use new detailed parser ***
    stats = parse_detailed_tripinfo(tripinfo_file)
    # if os.path.exists(tripinfo_file): os.remove(tripinfo_file) # <-- Kept for inspection
    return stats, queue_data

def run_model_simulation(chromosome, config_file):
    """Runs a model-controlled simulation and collects queue and trip data."""
    tripinfo_file = "tripinfo_test_model.xml"
    # Add --random if you want to test robustness,
    # or remove it to test on the exact same traffic as the baseline
    sumo_cmd = [SUMO_BINARY_TEST, "-c", config_file, "--tripinfo-output", tripinfo_file]
    
    traci.start(sumo_cmd)
    controller = ANN(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)
    controller.set_weights_from_flat(chromosome)
    
    next_green_phase_idx = 0
    queue_data = []
    while traci.simulation.getMinExpectedNumber() > 0:
        # --- Data Collection Step ---
        current_time = traci.simulation.getTime()
        halting_cars = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in INCOMING_LANES)
        queue_data.append((current_time, halting_cars))
        
        # --- Controller Logic Step ---
        current_phase = traci.trafficlight.getPhase(JUNCTION_ID)
        if current_phase in GREEN_PHASES:
            inputs = [traci.lane.getLastStepHaltingNumber(lane) for lane in INCOMING_LANES]
            green_duration = controller.predict(inputs)
            
            # Green Phase Logic
            start_time = traci.simulation.getTime()
            while traci.simulation.getTime() < start_time + green_duration:
                if traci.simulation.getMinExpectedNumber() == 0: break
                traci.simulationStep()
                # Also collect data during the green phase
                current_time = traci.simulation.getTime()
                halting_cars = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in INCOMING_LANES)
                queue_data.append((current_time, halting_cars))
            if traci.simulation.getMinExpectedNumber() == 0: break
            
            # Yellow Phase Logic
            yellow_phase = YELLOW_PHASES[current_phase]
            traci.trafficlight.setPhase(JUNCTION_ID, yellow_phase)
            start_time = traci.simulation.getTime()
            while traci.simulation.getTime() < start_time + YELLOW_PHASE_DURATION:
                if traci.simulation.getMinExpectedNumber() == 0: break
                traci.simulationStep()
                # Also collect data during the yellow phase
                current_time = traci.simulation.getTime()
                halting_cars = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in INCOMING_LANES)
                queue_data.append((current_time, halting_cars))
            if traci.simulation.getMinExpectedNumber() == 0: break
            
            # Switch to next phase
            next_green_phase_idx = 1 - next_green_phase_idx
            traci.trafficlight.setPhase(JUNCTION_ID, GREEN_PHASES[next_green_phase_idx])
        else:
            traci.simulationStep()
            
    traci.close()
    
    # *** MODIFIED: Use new detailed parser ***
    stats = parse_detailed_tripinfo(tripinfo_file)
    # if os.path.exists(tripinfo_file): os.remove(tripinfo_file) # <-- Kept for inspection
    return stats, queue_data


if __name__ == "__main__":
    print(f"--- Testing Model Performance on New Traffic Data ---")
    if not os.path.exists(SAVED_MODEL_FILE): sys.exit(f"Error: Model file '{SAVED_MODEL_FILE}' not found! Please run train_model.py first.")
    if not os.path.exists(TEST_CONFIG_FILE): sys.exit(f"Error: Test config file not found at '{TEST_CONFIG_FILE}'")

    # 1. Run Baseline Simulation
    print(f"\n[1/3] Running baseline simulation on '{os.path.basename(TEST_CONFIG_FILE)}'...")
    baseline_stats, baseline_queue_data = run_baseline_simulation(TEST_CONFIG_FILE)
    if baseline_stats['total_wait_time'] == float('inf'): sys.exit("Error during baseline simulation.")
    print(f"-> Baseline (Fixed Timer) Wait Time: {baseline_stats['total_wait_time']:.2f} seconds")

    # 2. Run Trained Model Simulation
    print(f"\n[2/3] Loading model and running simulation on '{os.path.basename(TEST_CONFIG_FILE)}'...")
    best_chromosome = np.load(SAVED_MODEL_FILE)
    print("-> Model weights loaded successfully.")
    model_stats, model_queue_data = run_model_simulation(best_chromosome, TEST_CONFIG_FILE)
    if model_stats['total_wait_time'] == float('inf'): sys.exit("Error during model simulation.")
    print(f"-> Trained Model (GA-ANN) Wait Time: {model_stats['total_wait_time']:.2f} seconds")
    
    # 3. Generate Plots
    print("\n[3/3] Generating all test plots...")
    plot_performance_comparison(baseline_stats, model_stats)
    plot_queue_length(baseline_queue_data, model_queue_data)
    plot_wait_time_histogram(baseline_stats, model_stats)
    plot_wait_time_boxplot(baseline_stats, model_stats)
    plot_distribution_histograms(baseline_stats, model_stats)

    # 4. Final Comparison
    print("\n--- Final Test Results (Test Data) ---")
    print(f"Metric                  | Baseline  | GA-ANN Model")
    print("-------------------------------------------------------")
    print(f"Total Waiting Time (s)  | {baseline_stats['total_wait_time']:<9.2f} | {model_stats['total_wait_time']:<9.2f}")
    print(f"Total Time Loss (s)     | {baseline_stats['total_time_loss']:<9.2f} | {model_stats['total_time_loss']:<9.2f}")
    print(f"Average Travel Time (s) | {baseline_stats['avg_travel_time']:<9.2f} | {model_stats['avg_travel_time']:<9.2f}")
    print(f"Vehicle Throughput      | {baseline_stats['throughput']:<9} | {model_stats['throughput']:<9}")
    
    if baseline_stats['total_wait_time'] > 0:
        improvement = ((baseline_stats['total_wait_time'] - model_stats['total_wait_time']) / baseline_stats['total_wait_time']) * 100
        print(f"\nImprovement in Waiting Time: {improvement:.2f}%")