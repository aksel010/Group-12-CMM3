# Group-12-CMM3 - Advanced Battery Thermal Management System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## Overview

This repository contains a Python-based computational framework for simulating and optimizing an innovative Battery Thermal Management System (BTMS) for Plug-in Hybrid Electric Vehicles (PHEVs). The system employs the vehicle's fuel (n-heptane) as a coolant medium, offering superior thermal performance compared to conventional water-glycol systems.

**Research Objectives:**
- Determine minimum feasible charging time while maintaining the battery temperature below 40°C
- Maximize charging current within thermal constraints

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
# cd Group-12-CMM3-1

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
virtualenv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
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

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| n_cell | 240 | - | total number of cells |
| v_cell | 1.2 | V | per cell |
| capacity_cell | 6 | Ah | per cell |
| dc_ir | 2.5e-3 | Ohm | per cell |
| m_cell | 0.158 | kg | test cell mass |
| m_b | m_cell * n_cell | kg | total mass |
| c_b | 2788 | J/kgK | specific heat capacity |
| q_b | capacity_cell * 3600 | As | total charge |
| r_b | dc_ir * n_cell | Ohm | total resistance |
| t_b_max | 40 + 273.13 | K | maximum safe temperature |
| t_in | 15 + 273.13 | K | inlet coolant temp |

### Channel Geometry

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| n | 5 | - | number of coolant branches |
| d | 17e-3 | m | main pipe diameter |
| w_branch | 30e-3 | m | branch width |
| h_branch | 2.75e-3 | m | branch height |
| l_a | 0.5 | m | main pipe segment length |
| l_b | 0.5 | m | branch length |
| l_c | 0.2 | m | branch outlet length |

### Operating Conditions

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| current_0 | 15 | A | initial current |
| current_threshold | 1e-6 | - | convergence threshold |
| mass_flow_initial | 0.0001 | kg/s | nominal mass flowrate |
| H | 30 | s | RK4 time integration step size |

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
