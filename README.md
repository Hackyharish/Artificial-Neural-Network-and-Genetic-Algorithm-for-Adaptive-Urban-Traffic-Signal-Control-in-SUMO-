# 🚦 Artificial Neural Network for Adaptive Urban Traffic Signal Control in SUMO

**Authors:** Harish R, Namitha Madhu
*Department of Electrical and Electronics Engineering, Amrita Vishwa Vidyapeetham, Coimbatore, India*

---

## 📄 Abstract

Urban cities face major consequences from traffic congestion and long waiting times at intersections. The conventionally used fixed-time traffic signals are inefficient since they cannot adapt dynamically to fluctuating traffic conditions.

This work presents a **Neuro-Evolutionary approach** that combines a **Genetic Algorithm (GA)** and an **Artificial Neural Network (ANN)** to optimize signal timings based on real-time traffic density.

* The **ANN** acts as the controller, dynamically allocating green times.
* The **GA** optimizes ANN weights to minimize total waiting time and delay.

The system is implemented in **Python** and simulated using **SUMO (Simulation of Urban Mobility)**.
Real-world data from a traffic junction in **Bremen, Germany**, was used for validation. The proposed GA-ANN system demonstrated a **79.31% improvement** in total waiting time over conventional fixed-time control.

**Keywords:** Neuroevolution, Genetic Algorithm, Artificial Neural Network, Traffic Signal Optimization, SUMO, Intelligent Transportation Systems

---

## 🧭 Methodology Overview

### 1️⃣ Simulation Environment

* **Tool:** SUMO (Simulation of Urban Mobility)
* **Interface:** Python – TraCI API
* **Dataset:** CN+ Vehicular Dataset (Bremen, Germany)

![Traffic signal location Bremen, Germany](map.png)

---

### 2️⃣ Flowchart

The process flow of the proposed GA-ANN-based optimization:

![Flow diagram](Flow_diagram.png)

---

## ⚙️ GA–ANN Model Design

### Artificial Neural Network (ANN)

| Parameter    | Description                           |
| ------------ | ------------------------------------- |
| Inputs       | Number of halting vehicles in 3 lanes |
| Hidden Layer | 10 neurons (sigmoid activation)       |
| Output       | Green time (scaled between 10s – 60s) |

### Genetic Algorithm (GA)

| Parameter       | Value                      |
| --------------- | -------------------------- |
| Population Size | 50                         |
| Generations     | 100                        |
| Crossover Rate  | 0.8                        |
| Mutation Rate   | 0.05                       |
| Selection       | Tournament (k=3)           |
| Elitism         | Top 1 individual preserved |

**Fitness Function:**

```latex
Fitness = \frac{1}{1 + W_{total}}
```

Where `W_total` = total vehicle waiting time per cycle.

---

## 🧩 System Architecture

```
📁 Project Folder
├── train_model.py
├── test_model.py
├── config/
│   ├── CN+ Dataset.sumocfg
│   ├── network.net.xml
│   ├── routes.rou.xml
│   └── additional.add.xml
├── results/
│   ├── best_ann_weights.npy
│   ├── ga_convergence_plot.png
│   ├── test_performance_comparison.png
│   ├── wait_time_boxplot.png
│   ├── wait_time_histogram.png
│   ├── tripinfo_baseline.xml
│   └── tripinfo_ga_run.xml
└── README.md
```

---

## 🧠 How to Run the Code

### ⚙️ 1. Prerequisites

#### Install SUMO

* Download from: [https://www.eclipse.org/sumo](https://www.eclipse.org/sumo)
* Add to your system PATH:

```bash
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
```

#### Install Python Libraries

```bash
pip install numpy matplotlib
```

#### Verify SUMO

```bash
echo %SUMO_HOME%
```

Ensure it prints your SUMO installation path.

---

### 🧩 2. Training the Model

**File:** `train_model.py`

This script trains the ANN using GA to minimize vehicle waiting time.

#### ✅ Steps:

1. Update your SUMO configuration path:

   ```python
   CONFIG_FILE = r"D:\Path\to\CN+ Dataset\SUMO Format\Infrastructure Files\CN+ Dataset.sumocfg"
   ```
2. Run the script:

   ```bash
   python Train.py
   ```
3. The script will:

   * Run a baseline SUMO simulation
   * Evolve ANN weights with GA
   * Save the best ANN as `best_ann_weights.npy`
   * Plot convergence graphs and comparisons

#### 📁 Output Files

| File                                            | Description            |
| ----------------------------------------------- | ---------------------- |
| `best_ann_weights.npy`                          | Optimized ANN weights  |
| `ga_convergence_plot.png`                       | GA fitness evolution   |
| `summary_bar_chart.png`                         | Performance comparison |
| `tripinfo_baseline.xml` / `tripinfo_ga_run.xml` | SUMO trip statistics   |

---

### 🧪 3. Testing the Model

**File:** `test_model.py`

This evaluates the optimized ANN controller in a real-time SUMO simulation.

#### ✅ Steps:

1. Update SUMO test config path:

   ```python
   TEST_CONFIG_FILE = r"C:\Path\to\CN+ Dataset\SUMO Format\Infrastructure Files\CN+ Dataset.sumocfg"
   ```
2. Run with GUI enabled:

   ```bash
   python Test.py
   ```
3. To speed up (without GUI):

   ```bash
   python Test.py --nogui
   ```

#### 🧾 Output:

* SUMO visual simulation
* Metrics: average waiting time, travel time, time loss
* Plots generated in `/results`

---

### ⚠️ Notes

* Ensure `train_model.py`, `test_model.py`, and SUMO config files are in the same directory.
* Use **raw strings (r"path")** for Windows paths.
* Close SUMO GUI before re-running to avoid TraCI port conflicts.

---

## 📊 Results and Discussion

| Metric             | Fixed-Time | GA–ANN     | Improvement  |
| ------------------ | ---------- | ---------- | ------------ |
| Total Waiting Time | 20015 s    | 4141 s     | **79.31% ↓** |
| Total Time Loss    | 27998.98 s | 10483.07 s | **62.55% ↓** |
| Avg. Travel Time   | 69.80 s    | 40.49 s    | **41.99% ↓** |
| Vehicle Throughput | 590        | 598        | **↑ 1.35%**  |

---

### 🔁 GA Convergence

![GA Convergence](ga_convergence_plot.png)

### 🚗 Queue Length

![Queue Length](test_queue_length.png)

### ⏱ Wait Time Distributions

![Wait Time Distribution](test_time_distributions.png)
![Box Plot](test_wait_time_boxplot.png)

### 🧍‍♂️ Travel Time & Time Loss

![Travel Time and Time Loss](test_wait_time_histogram.png)

---

## 🏁 Conclusion

This project developed an **adaptive traffic signal controller** using a hybrid **Genetic Algorithm–Artificial Neural Network** model.
It achieved a **79.31% reduction** in total waiting time and significant improvements in throughput and average travel time.
The approach proves that **bio-inspired optimization** can effectively enhance traffic management in urban intersections.

---

## 🚀 Future Enhancements

1. **Multi-Intersection Control** for networked traffic signals.
2. Integration of **real-time traffic cameras** or **IoT sensors**.
3. Comparison with **Reinforcement Learning (DQN, PPO)** controllers.
4. **Multi-Objective GA** for optimizing fuel efficiency and emissions.

---

## 🧩 Citation

If you use this work, please cite as:

> Harish R, Namitha Madhu, *“Artificial Neural Network for Adaptive Urban Traffic Signal Control in SUMO,”* Department of Electrical and Electronics Engineering, Amrita Vishwa Vidyapeetham, Coimbatore, India, 2025.

---

## 📧 Contact

**Harish R**
Department of EEE, Amrita Vishwa Vidyapeetham, Coimbatore, India
Email: [harish@example.com](mailto:harishr.vnr@gmail.com)
LinkedIn: [https://www.linkedin.com/in/harish-r](https://www.linkedin.com/in/harish-r-8b68a333b/)
GitHub: [https://github.com/harish-r](https://github.com/harish-r)
**Namitha Madhu**
Department of EEE, Amrita Vishwa Vidyapeetham, Coimbatore, India
Email: [harish@example.com](mailto:cb.en.u4eee23149@cb.students.amrita.edu)
LinkedIn: [https://www.linkedin.com/in/harish-r](https://www.linkedin.com/in/namitha-madhu-4934a8276/)
GitHub: [https://github.com/harish-r](https://github.com/harish-r)
