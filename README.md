# Group-12-CMM3 - Advanced Battery Thermal Management System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## Overview

This repository contains a comprehensive computational framework for simulating and optimizing an innovative **Battery Thermal Management System (BTMS)** for Plug-in Hybrid Electric Vehicles (PHEVs). The system uniquely employs the vehicle's fuel (n-heptane) as a coolant medium, offering superior thermal performance and energy efficiency compared to conventional water-glycol systems.

### Research Objectives

- Determine **minimum feasible charging time** while maintaining battery temperature below 40°C
- **Maximize charging current** within thermal safety constraints
- Analyze system sensitivity to operating parameters (mass flow rate, inlet temperature, RK4 step size)
- Provide practical **C-rate recommendations** for real-world charging scenarios

## Key Features

🔬 **Multi-Physics Modeling**
- Thermal dynamics with transient temperature evolution
- Thermohydraulic analysis with realistic fluid dynamics
- Electrochemical heating from battery resistance losses

⚡ **Advanced Numerical Methods**
- **RK4 integration** for ODE solving with automatic step-size control
- **Cubic spline interpolation** for smooth parameter curves
- **Newton-Raphson root finding** for optimization
- **Monte Carlo error propagation** for uncertainty quantification

🎯 **Optimization Engine**
- Iterative convergence algorithm for critical current determination
- Automated sensitivity analysis across parameter space
- Parallel processing support for batch studies

🖥️ **User Interfaces**
- **Interactive GUI** with real-time visualization
- **Command-line interface** for scripting and automation
- **Python API** for programmatic access

📊 **Data & Validation**
- NIST-validated thermophysical property correlations for n-heptane
- Real charging profile integration
- Comprehensive result logging and visualization

## Project Structure

```
Group-12-CMM3/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies (exact versions)
├── setup.py                               # Package configuration (minimum versions)
├── .gitignore                             # Git exclusions
│
├── data/                                  # Input thermophysical data
│   ├── raw/                               # Original data sources
│   │   ├── n_heptane_2.txt               # NIST n-heptane properties
│   │   └── Real_Data_C_Rate.csv          # Real charging data
│   └── processed/                         # Fitted coefficients and processed data
│       ├── TempTimeCharging_LIU2.csv
│       └── time_of_simulation.csv
│
├── src/                                   # Source code package
│   ├── __init__.py
│   ├── config.py                          # Global configuration parameters
│   │
│   ├── models/                            # Core physics simulation modules
│   │   ├── __init__.py
│   │   ├── battery_temperature_ode.py     # Battery temp evolution (RK4 & SciPy)
│   │   ├── mass_flowrate.py               # Thermohydraulic solver
│   │   ├── optimum_current.py             # Current optimization algorithm
│   │   ├── rk4_error.py                   # RK4 truncation error analysis
│   │   └── cooling_analysis.py            # Channel head-loss model
│   │
│   ├── analysis/                          # Sensitivity and parametric studies
│   │   ├── __init__.py
│   │   ├── time_sensitivity.py            # Step-size sensitivity
│   │   ├── current_sensitivity.py         # Current parameter study
│   │   └── massflow_sensitivity.py        # Mass flow parameter study
│   │
│   ├── utils/                             # Utility and helper functions
│   │   ├── __init__.py
│   │   ├── heptane_interpolater.py        # n-heptane property correlations
│   │   ├── interpolater.py                # Spline and interpolation routines
│   │   ├── root_finders.py                # Newton & bisection methods
│   │   ├── c_rate_interpolation.py        # C-rate calculation utilities
│   │   └── clear_values.py                # Data reset utilities
│   │
│   └── visualisation/                     # Plotting and visualization
│       ├── __init__.py
│       └── comparison_plot.py             # Multi-plot generation
│
├── scripts/                               # Executable entry points
│   ├── main.py                            # Full analysis workflow
│   ├── gui.py                             # Interactive GUI launcher
│   └── charging_time_analysis.py          # Charging performance calculator
│
├── results/                               # Generated outputs (not tracked)
│   ├── figures/                           # Plots and visualizations
│   ├── tables/                            # CSV and summary data
│   └── validation/                        # Sensitivity study results
│
└── key files/                             # Reference documentation
```

## Installation & Setup

### Prerequisites

- **Python 3.8+** (tested on 3.9, 3.10, 3.11, 3.13)
- **pip** package manager
- **Git** for version control
- ~500 MB disk space for dependencies

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd Group-12-CMM3

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install in editable mode (recommended for development)
pip install -e .

# Or install with pinned versions (for reproducibility)
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test imports
python -c "import src.models.battery_temperature_ode; print('✓ Installation successful')"
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Numerical computing |
| scipy | ≥1.10.0 | Scientific algorithms & ODE solving |
| matplotlib | ≥3.7.0 | Visualization & plotting |
| pandas | ≥2.0.0 | Data manipulation |
| seaborn | ≥0.12.0 | Statistical visualization |
| scikit-learn | ≥1.3.0 | Machine learning (sensitivity analysis) |

## Usage Guide

### Option 1: Interactive GUI

```bash
python scripts/gui.py
```

**Features:**
- Adjust parameters in real-time (current, step size, temperatures)
- Monitor convergence progress with live logging
- View generated plots immediately
- Export results to CSV

**Workflow:**
1. Enter desired charging current (1-17.5 A)
2. Set RK4 step size (10-60 s)
3. Configure thermal limits (35-50°C battery, 10-25°C coolant inlet)
4. Click "Run Complete Analysis"
5. View results, C-rate recommendation, and plots

### Option 2: Command-Line Interface

```bash
python scripts/main.py
```

**Outputs to console:**
- Converged optimum current value
- RK4 error analysis
- Mass flowrate calculations
- ODE solution comparison (RK4 vs SciPy)
- Real charging time estimates
- Generated plots saved to `results/figures/`

### Option 3: Python API

```python
from src.config import current_store, current_threshold
import src.models.optimum_current as oc
import src.models.battery_temperature_ode as ode
from src.models.optimum_current import current_params

# Configure parameters
import src.config as config
config.current_0 = 15.0      # Starting current (A)
config.H = 30                # RK4 step size (s)
config.t_b_max = 313.13      # Max battery temp (K)
config.t_in = 288.13         # Inlet coolant temp (K)

# Run convergence loop
current_store.clear()
for iteration in range(10):
    result = oc.run()
    critical_current = result['critical'][0]
    current_store.append(critical_current)
    
    # Check convergence
    if len(current_store) > 1:
        error = abs(current_store[-1] - current_store[-2]) / current_store[-1]
        if error < current_threshold:
            print(f"✓ Converged: {critical_current:.4f} A")
            break

# Solve ODE with optimal current
from src.models.battery_temperature_ode import d_tb_dt, get_tb, get_tb_scipy
params = current_params(current_store[-1])
time_rk4, temp_rk4 = get_tb(d_tb_dt, params, stepsize=config.H)
time_scipy, temp_scipy = get_tb_scipy(d_tb_dt, params)

print(f"Battery reaches {max(temp_rk4):.2f} K")
```

## Configuration

All parameters are configured in `src/config.py`. Key tunable parameters:

### Battery Parameters
```python
capacity_cell = 6          # Individual cell capacity (Ah)
n_cell = 240               # Number of cells in series
t_b_max = 313.13           # Max battery temperature (K, ~40°C)
dc_ir = 2.5e-3             # Internal resistance per cell (Ω)
```

### Thermal Management
```python
t_in = 288.13              # Inlet coolant temperature (K, ~15°C)
mass_flow_initial = 0.0001 # Initial mass flow guess (kg/s)
H = 30                     # RK4 integration step size (s)
```

### Cooling System
```python
n = 5                      # Number of parallel branches
d = 17e-3                  # Main pipe diameter (m)
w_branch = 30e-3           # Channel width (m)
h_branch = 2.75e-3         # Channel height (m)
```

### Convergence Control
```python
current_threshold = 1e-6   # Relative convergence tolerance
```

## Understanding the Workflow

### 1. **Current Optimization Loop**
```
Initialize current → Solve ODE → Calculate ΔT → 
  Check if ΔT ≈ 0 → 
    Yes: Converged ✓ 
    No: Update current → Repeat
```

### 2. **Temperature Evolution (ODE Solver)**
```
dT_b/dt = [I²R - h·A_s·(T_b - T_c)] / (m·c_b)

Where:
  I = charging current
  R = battery resistance
  h = convective heat transfer coefficient
  A_s = surface area
  T_c = coolant temperature
```

### 3. **Mass Flow Rate Calculation**
Solves coupled thermohydraulic equations for steady-state coolant flow.

### 4. **C-Rate Calculation**
```
C-rate = I / Q_nominal
  where Q = battery capacity in Ah

Example: 6Ah battery with 15A current = 2.5C charging rate
```

## Output Files

Results are automatically saved to `results/`:

```
results/
├── figures/
│   ├── gui_analysis_plot.png          # Main 3-panel plot from GUI
│   └── main_plot.png                  # Full analysis visualization
├── tables/
│   ├── gui_summary.txt                # GUI session results
│   ├── main_summary.txt               # Full analysis summary
│   └── RK4_solution.csv               # Detailed ODE solution data
└── validation/
    ├── time_sensitivity_summary.txt   # Step-size study results
    ├── current_sensitivity.csv        # Current parameter sweep
    └── massflow_sensitivity.csv       # Flow rate parameter sweep
```

## Troubleshooting

### "Module not found" errors
```bash
# Ensure package is installed
pip install -e .
# Or add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Group-12-CMM3"
```

### GUI doesn't start
```bash
# Check tkinter is available (comes with Python)
python -m tkinter
# If blank window opens, tkinter is working

# Try running from project root
cd /path/to/Group-12-CMM3
python scripts/gui.py
```

### Convergence fails or takes too long
- Try increasing `current_threshold` (e.g., 1e-5 instead of 1e-6)
- Reduce initial current range in `optimum_current.py` (line 160)
- Increase `H` (RK4 step size) for faster but less accurate solutions

### Results don't match between GUI and main.py
- Ensure all parameters are set identically
- Check `config.py` hasn't been modified
- GUI input validation: see range warnings in status messages

## Performance Benchmarks

| Task | Typical Time | Notes |
|------|-------------|-------|
| Single ODE solve | 2-5 s | Depends on stepsize (H) |
| Optimum current convergence | 60-180 s | Usually 3-5 iterations |
| Full analysis (main.py) | 180-300 s | Includes all modules + plotting |
| GUI startup | <2 s | Lazy loading of computation modules |
| Sensitivity study | 10-20 min | 100+ parameter combinations |

## Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Code Style
- Follow PEP 8
- Use type hints where practical
- Add docstrings to all functions

### Adding Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and test locally
3. Commit with clear message: `git commit -m "feat: description"`
4. Push and create pull request

## References

1. **Al Qubeissi, M., Almshahy, A., et al.** (2022). "Modelling of battery thermal management: A new concept of cooling using fuel." *Fuel*, 310, 122456. [DOI](https://doi.org/10.1016/j.fuel.2021.122456)

2. **NIST Chemistry WebBook.** Thermophysical Properties of n-Heptane. https://webbook.nist.gov/cgi/cbook.cgi?ID=C142825

3. **Incropera, F. P., & DeWitt, D. P.** (2007). *Fundamentals of Heat and Mass Transfer* (6th ed.). John Wiley & Sons.

4. **White, F. M.** (2011). *Fluid Mechanics* (7th ed.). McGraw-Hill.

5. **Sharma, R., et al.** (2019). "A review on electro-thermal modelling and management of lithium-ion batteries." *Journal of Energy Storage*, 21, 713-737.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{group12_btms_2025,
  title={Group-12-CMM3: Advanced Battery Thermal Management System},
  author={Group 12},
  year={2025},
  institution={University of Edinburgh},
  url={https://github.com/aksel010/Group-12-CMM3}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support & Contact

For issues, questions, or suggestions:
- **GitHub Issues**: [Create an issue](../../issues)
- **Email**: [Group 12 - University of Edinburgh]

---

**🔬 Built at the University of Edinburgh | 2025**

*Advancing electric vehicle battery technology through computational innovation*

## Key Features

- Multi-physics coupling: thermal, fluid dynamics, and electrochemical models
- Transient analysis with multi-node spatial discretization
- Automated optimization using parallel processing
- Interactive GUI and command-line interfaces
- NIST-validated thermophysical property correlations

## Project Structure

```
Group-12-CMM3-1/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Version control exclusions
│
├── data/                          # Thermophysical data
│   ├── raw/                       # Original NIST data
│   │   ├── n_heptane_2.txt      
│   │   ├── real_data_c_rate.csv   
│   └── processed/                 # Fitted polynomial coefficients
│       ├── temp_time_charging_liu2               
│       └── time_of_simulation.csv         
│
├── src/                           # Source code
│   ├── models/                    # Core physics models
│   │   ├── cooling_analysis.py    # Cooling channel network head-loss model
│   │   ├── Mass_flowrate.py       # Thermohydraulic coolant loop and mass flow rate calculation
│   │   ├── ODE.py                 # Battery temperature evolution using RK4 and SciPy
│   │   ├── Optimum_Current.py     # Battery charging current optimization
│   │   └── RK4_Error.py           # RK4 error analysis
│   │
│   ├── analysis/                  # Sensitivity studies
│   │   ├── mass_flow_rate_current_sensitivity_anaylysis.py
│   │   ├── sensitivity_T_vary_mdot.py
│   │   └── time_sensitivity_analysis.py
│   │
│   ├── utils/                     # Helper functions
│   │   ├── c_rate_real_interpolation.py
│   │   ├── clear_values.py
│   │   ├── heptane_itpl.py        # n-heptane thermophysical properties
│   │   ├── interpolater.py
│   │   └── root_finders.py
│   │
│   ├── visualisation/             # Plotting functions
│   │   └── comparison_plot.py
│   │
│   ├── config.py                  # Configuration parameters
│   └── gui_app.py                 # Graphical interface
│
├── scripts/                       # Executable entry points
│   ├── main.py                    # Main execution script
│   └── Real_Charging_Time.py
│
├── tests/                         # Unit tests and validation
│
├── docs/                          # Documentation
│
└── results/                       # Generated outputs (git-ignored)
    ├── figures/
    └── tables/
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git version control

### Setup Instructions

```bash
# Clone repository
# git clone <repository-url>
# cd Group-12-CMM3

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode
# This makes the 'src' package available to all scripts
pip install -e .
```

### Dependencies

```
numpy>=1.21.0      # Numerical computing
scipy>=1.7.0       # Scientific computing and optimization
matplotlib>=3.5.0  # Visualization
pandas>=1.3.0      # Data manipulation
```

## Usage

### Graphical User Interface

```bash
python src/gui_app.py
```

The GUI provides:

- Parameter configuration interface
- Real-time simulation progress
- Interactive result visualization

### Command-Line Interface

```bash
python scripts/main.py
```

Outputs optimal design parameters and plots:

- Optimum Current (A)
- ODE Solution Validation
- Pressure Balance Residual

### Python API

```python
import src.models.ODE as ODE
import src.models.Mass_flowrate as mf
import src.models.Optimum_Current as oc

# Run the full analysis
if __name__ == "__main__":
    # Run convergence loop for optimum current
    current_store_result = compute_optimum_current(threshold=1e-6)
    print(f'\n✓ Model converged after {len(current_store_result)} iterations')
    print(f'Optimum Current: {current_store_result[-1]:.4f} A\n')
    print("==== Group 12 - CMM3 Consolidated Results ====\n")
    ode_data = ODE.run()
    print("\n--- Mass Flowrate Solver ---")
    mf_data = mf.run()
    print("\n--- Optimum Current Analysis ---")
    oc_data = oc.run()
```

## Configuration Parameters

### Battery Specifications

| Parameter     | Default              | Units | Description              |
| ------------- | -------------------- | ----- | ------------------------ |
| n_cell        | 240                  | -     | total number of cells    |
| v_cell        | 1.2                  | V     | per cell                 |
| capacity_cell | 6                    | Ah    | per cell                 |
| dc_ir         | 2.5e-3               | Ohm   | per cell                 |
| m_cell        | 0.158                | kg    | test cell mass           |
| m_b           | m_cell * n_cell      | kg    | total mass               |
| c_b           | 2788                 | J/kgK | specific heat capacity   |
| q_b           | capacity_cell * 3600 | As    | total charge             |
| r_b           | dc_ir * n_cell       | Ohm   | total resistance         |
| t_b_max       | 40 + 273.13          | K     | maximum safe temperature |
| t_in          | 15 + 273.13          | K     | inlet coolant temp       |

### Channel Geometry

| Parameter | Default | Units | Description                |
| --------- | ------- | ----- | -------------------------- |
| n         | 5       | -     | number of coolant branches |
| d         | 17e-3   | m     | main pipe diameter         |
| w_branch  | 30e-3   | m     | branch width               |
| h_branch  | 2.75e-3 | m     | branch height              |
| l_a       | 0.5     | m     | main pipe segment length   |
| l_b       | 0.5     | m     | branch length              |
| l_c       | 0.2     | m     | branch outlet length       |

### Operating Conditions

| Parameter         | Default | Units | Description                    |
| ----------------- | ------- | ----- | ------------------------------ |
| current_0         | 15      | A     | initial current                |
| current_threshold | 1e-6    | -     | convergence threshold          |
| mass_flow_initial | 0.0001  | kg/s  | nominal mass flowrate          |
| H                 | 30      | s     | RK4 time integration step size |

## Development Team

**University of Edinburgh**

This project was developed by a team of students at the University of Edinburgh.

## References

1. Al Qubeissi, M., Almshahy, A., et al. "Modelling of battery thermal management: A new concept of cooling using fuel." *Fuel*, vol. 310, 2022.
2. NIST Chemistry WebBook. "n-Heptane Thermophysical Properties." https://webbook.nist.gov/cgi/cbook.cgi?ID=C142825
3. Incropera, F. P., & DeWitt, D. P. *Fundamentals of Heat and Mass Transfer* (6th ed.). John Wiley & Sons, 2007.
4. White, F. M. *Fluid Mechanics* (7th ed.). McGraw-Hill, 2011.

---

**Built at the University of Edinburgh | 2025**
