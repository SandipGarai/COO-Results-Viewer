# COO & ANN Results Viewer

A comprehensive Streamlit-based visualization and statistical analysis tool for evaluating the **Canine Olfactory Optimization Algorithm (COO)** against baseline metaheuristic optimizers on benchmark functions and Artificial Neural Network (ANN) hyperparameter optimization tasks.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Statistical Methods](#statistical-methods)
- [Algorithms Compared](#algorithms-compared)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Citation](#citation)
- [Authors](#authors)

---

## Overview

This application provides a unified platform for analyzing optimization algorithm performance across two experimental domains:

1. **Benchmark Function Optimization**: Evaluating COO on standard mathematical test functions (Sphere, Rastrigin, Ackley, etc.) with various transformations (Shifted, Rotated, ShiftedRotated).

2. **ANN Hyperparameter Optimization**: Comparing optimizers for tuning neural network hyperparameters across multiple datasets (synthetic, Boston housing, California housing, diabetes prediction).

---

## Features

### Visualization Capabilities

- **Convergence History Plots**: Track optimization progress across iterations with log/linear scale options
- **3D Surface Plots**: Interactive visualization of objective function landscapes
- **Box Plots**: Distribution analysis across multiple seeds
- **Bar Charts**: Comparative performance metrics

### Statistical Analysis

- **Wilcoxon Signed-Rank Test**: Non-parametric pairwise comparisons
- **Cohen's d Effect Size**: Practical significance measurement
- **Win/Tie/Loss Analysis**: Head-to-head algorithm comparisons
- **Average Ranking**: Overall algorithm ordering

### Data Processing

- **Automatic Outlier Filtering**: Removes extreme penalty values (e.g., 1e12) from convergence histories
- **Absolute Value Conversion**: Ensures log-scale compatibility
- **Multi-seed Aggregation**: Statistical summaries across random seeds

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/SandipGarai/COO-Results-Viewer.git
cd coo-ann-viewer
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## Usage

### Directory Structure Requirements

Place your data folders in the same directory as the application:

```
project_root/
├── app.py
├── requirements.txt
├── README.md
├── Functions_YYYYMMDD_HHMMSS/          # Benchmark function results
│   ├── 2D/
│   │   └── [FunctionName]/
│   │       └── COO/
│   │           └── [Variant]/
│   │               └── plotly/
│   │                   └── *.html
│   ├── optuna_tuning_results/
│   │   └── trials_*.csv
│   └── benchmark_master_final.csv
│
└── ANN_YYYYMMDD_HHMMSS/                # ANN optimization results
    └── benchmark_master.csv
```

### Navigation

1. **Section Selection**: Choose between "Functions" or "ANN" in the sidebar
2. **Page Selection**: Navigate to specific analysis pages
3. **Filter Controls**: Select optimizers, datasets, seeds, and metrics
4. **View Options**: Toggle between different visualization modes

---

## Data Format

### Benchmark Functions CSV (`benchmark_master_final.csv`)

| Column       | Type    | Description                                                |
| ------------ | ------- | ---------------------------------------------------------- |
| `optimizer`  | string  | Algorithm name (COO, PSO, DE, GWO, WOA)                    |
| `function`   | string  | Function name with variant (e.g., Sphere_Shifted)          |
| `seed`       | integer | Random seed for reproducibility                            |
| `best_value` | float   | Best objective value found                                 |
| `time_sec`   | float   | Execution time in seconds                                  |
| `conv_hist`  | string  | Convergence history as list (e.g., "[1.2, 0.8, 0.3, ...]") |

### ANN Benchmark CSV (`benchmark_master.csv`)

| Column       | Type    | Description                                            |
| ------------ | ------- | ------------------------------------------------------ |
| `optimizer`  | string  | Algorithm name                                         |
| `dataset`    | string  | Dataset name (synthetic, boston, california, diabetes) |
| `seed`       | integer | Random seed                                            |
| `mse`        | float   | Mean Squared Error achieved                            |
| `time_sec`   | float   | Execution time in seconds                              |
| `evals`      | integer | Number of function evaluations                         |
| `status`     | string  | Optimization status                                    |
| `best_pos`   | string  | Best position vector (10-dimensional)                  |
| `conv_hist`  | string  | Convergence history                                    |
| `opt_params` | string  | Optimizer-specific parameters                          |

### ANN Hyperparameter Encoding

The 10-dimensional search space for ANN hyperparameters:

| Index | Parameter             | Range                  | Decoding Formula        |
| ----- | --------------------- | ---------------------- | ----------------------- |
| 0     | Hidden Layer 1 Size   | [8, 256]               | h₁ = 8 + x₀ × (256 - 8) |
| 1     | Hidden Layer 2 Size   | [8, 256]               | h₂ = 8 + x₁ × (256 - 8) |
| 2     | L2 Regularization (α) | [1e-6, 0.1]            | α = 10^(-6 + 5x₂)       |
| 3     | Learning Rate         | [1e-5, 0.03]           | η = 10^(-5 + 3.5x₃)     |
| 4     | Activation Function   | {relu, tanh, logistic} | Categorical mapping     |
| 5     | Number of Layers      | {1, 2, 3}              | Categorical mapping     |
| 6     | Dropout Rate          | [0, 0.5]               | d = 0.5 × x₆            |
| 7     | Batch Size            | {16, 32, 64, 128}      | Categorical mapping     |
| 8     | Optimizer             | {sgd, adam, lbfgs}     | Categorical mapping     |
| 9     | Skip Connection       | {False, True}          | x₉ > 0.5                |

---

## Statistical Methods

### Wilcoxon Signed-Rank Test

A non-parametric test for comparing paired samples without assuming normal distribution.

**Hypotheses:**

- **H₀**: The median difference between pairs is zero
- **H₁**: The baseline algorithm performs better (one-sided)

**Test Statistic:**

$$W = \sum_{i=1}^{n} \text{sgn}(x_{2,i} - x_{1,i}) \cdot R_i$$

where Rᵢ is the rank of |x₂,ᵢ - x₁,ᵢ|.

**Significance Level:** α = 0.05

### Cohen's d Effect Size

Measures the standardized difference between two means:

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$$

where the pooled standard deviation is:

$$s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

**Interpretation:**

| Absolute d Value | Interpretation |
| ---------------- | -------------- |
| < 0.2            | Negligible     |
| 0.2 - 0.5        | Small          |
| 0.5 - 0.8        | Medium         |
| ≥ 0.8            | Large          |

**Sign Interpretation:**

- **d < 0**: Baseline performs better
- **d > 0**: Competitor performs better

### Efficiency Score

Combines solution quality with computational cost.

#### For Benchmark Functions:

$$\text{Efficiency Score} = f_{best} \times t_{exec}$$

where:

- $f_{best}$ = Best objective value found
- $t_{exec}$ = Execution time in seconds

#### For ANN Hyperparameter Optimization:

**Efficiency (MSE × Time):**

$$\text{Efficiency}_{time} = \text{MSE} \times t_{exec}$$

**Efficiency (MSE × Evaluations):**

$$\text{Efficiency}_{evals} = \text{MSE} \times n_{evals}$$

where:

- $\text{MSE}$ = Mean Squared Error achieved
- $t_{exec}$ = Execution time in seconds
- $n_{evals}$ = Number of function evaluations

**Interpretation:**

- **Lower efficiency score = Better performance** (fast and accurate)
- Rewards algorithms that find good solutions quickly
- Direct multiplication ensures both quality and speed are considered

### Average Ranking

For each function-seed (or dataset-seed) combination, algorithms are ranked 1 to k based on performance. The average rank is:

$$\bar{R}_j = \frac{1}{N} \sum_{i=1}^{N} r_{ij}$$

where $r_{ij}$ is the rank of algorithm j on problem instance i.

### Win/Tie/Loss Analysis

For baseline algorithm A vs competitor B:

- **Win**: $f_A < f_B$ (baseline is better)
- **Tie**: $f_A = f_B$ (equal performance)
- **Loss**: $f_A > f_B$ (competitor is better)

**Win Rate:**

$$\text{Win Rate} = \frac{\text{Wins}}{\text{Wins} + \text{Losses}} \times 100\%$$

---

## Algorithms Compared

### Canine Olfactory Optimization Algorithm (COO)

A novel nature-inspired metaheuristic based on the olfactory search and social behaviors of canines.

### Baseline Algorithms

| Algorithm | Full Name                    | Reference                 |
| --------- | ---------------------------- | ------------------------- |
| **PSO**   | Particle Swarm Optimization  | Kennedy & Eberhart (1995) |
| **DE**    | Differential Evolution       | Storn & Price (1997)      |
| **GWO**   | Grey Wolf Optimizer          | Mirjalili et al. (2014)   |
| **WOA**   | Whale Optimization Algorithm | Mirjalili & Lewis (2016)  |

### Benchmark Functions

| Function       | Formula                                                   | Global Minimum |
| -------------- | --------------------------------------------------------- | -------------- |
| **Sphere**     | f(x) = Σᵢ xᵢ²                                             | f(0) = 0       |
| **Rastrigin**  | f(x) = 10n + Σᵢ [xᵢ² - 10cos(2πxᵢ)]                       | f(0) = 0       |
| **Ackley**     | f(x) = -20exp(-0.2√(Σxᵢ²/n)) - exp(Σcos(2πxᵢ)/n) + 20 + e | f(0) = 0       |
| **Rosenbrock** | f(x) = Σᵢ [100(xᵢ₊₁ - xᵢ²)² + (1-xᵢ)²]                    | f(1) = 0       |
| **Griewank**   | f(x) = Σxᵢ²/4000 - Πcos(xᵢ/√i) + 1                        | f(0) = 0       |

### Function Transformations

- **Base**: Original function
- **Shifted**: f(x - o) where o is a shift vector
- **Rotated**: f(Mx) where M is a rotation matrix
- **ShiftedRotated**: f(M(x - o))

---

## Project Structure

```
coo-ann-viewer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This documentation file
├── Functions_*/           # Benchmark function results (auto-detected)
└── ANN_*/                 # ANN optimization results (auto-detected)
```

### Key Components in `app.py`

| Component                       | Description                                       |
| ------------------------------- | ------------------------------------------------- |
| `parse_conv_hist()`             | Parses convergence history with outlier filtering |
| `filter_extreme_outliers()`     | IQR-based outlier detection and replacement       |
| `cohens_d()`                    | Cohen's d effect size calculation                 |
| `perform_wilcoxon_test()`       | Wilcoxon signed-rank test implementation          |
| `calculate_rankings()`          | Average ranking computation                       |
| `win_tie_loss_analysis()`       | W/T/L comparison analysis                         |
| `apply_publication_style()`     | Publication-ready plot formatting                 |
| `decode_vector_to_ann_params()` | ANN hyperparameter decoding                       |

---

## Screenshots

### ANN Convergence History

- Visualize MSE convergence across iterations
- Compare multiple optimizers simultaneously
- Log scale for better visualization

### Statistical Analysis

- Comprehensive pairwise comparisons
- Effect size visualization
- Win/Tie/Loss bar charts

### 3D Function Visualization

- Interactive surface plots
- Optimization trajectory overlay
- Multiple seed comparison

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{COO_Results_viewer,
  author = {Garai, Sandip and Kanaka, K K},
  title = {COO & ANN Results Viewer: A Comprehensive Optimization Analysis Tool},
  year = {2026},
  url = {https://github.com/SandipGarai/COO-Results-Viewer}
}
```

---

## Authors

- [Dr. Sandip Garai](https://iiab.icar.gov.in/all-staff.php?view=025776#gsc.tab=0)
- [Dr. Kanaka K K](https://iiab.icar.gov.in/all-staff.php?view=025817#gsc.tab=0)

[📧 Contact](mailto:drgaraislab@gmail.com)

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://mit-license.org/) file for details.

---

## Acknowledgments

- Streamlit team for the excellent web framework
- Plotly for interactive visualization capabilities
- SciPy for statistical testing functions

---

**COO Results Viewer v1.0** | Developed for academic research in metaheuristic optimization
