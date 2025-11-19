# Group-12-CMM3: Plug-in Hybrid Electric Vehicle Battery Thermal Management System Optimization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## Abstract

This repository contains a comprehensive computational framework for the simulation and optimization of an advanced Battery Thermal Management System (BTMS) for Plug-in Hybrid Electric Vehicles (PHEVs). The system employs n-heptane, a readily available hydrocarbon fuel, as a thermal coupling medium between the battery pack and a vehicle heat recovery circuit. Through multi-physics coupling of thermodynamics, fluid mechanics, and electrochemical modeling, the framework determines optimal charging rates that minimise charging time while maintaining strict thermal safety constraints. This work demonstrates the feasibility of fuel-based thermal management as an alternative to conventional water-glycol systems, with implications for vehicle thermal architecture and charging infrastructure design.

## 1. Project Structure and Architecture

```
Group-12-CMM3/
│
├── README.md                              # User guide (this document)
├── requirements.txt                       # Pinned dependency versions
├── setup.py                               # Package configuration
├── .gitignore                             # Git exclusion rules
│
├── data/                                  # Experimental and reference data
│   ├── raw/                               # Primary data sources
│   │   ├── n_heptane_2.txt               # NIST thermophysical properties
│   │   └── Real_Data_C_Rate.csv          # Real-world charging profiles
│   └── processed/                         # Fitted coefficients
│       ├── TempTimeCharging_LIU2.csv
│       └── time_of_simulation.csv
│
├── src/                                   # Primary source code package
│   ├── __init__.py
│   ├── config.py                          # Global configuration constants
│   │
│   ├── models/                            # Core physics modules
│   │   ├── __init__.py
│   │   ├── battery_temperature_ode.py     # ODE solver (RK4 & SciPy)
│   │   ├── mass_flowrate.py               # Thermohydraulic analysis
│   │   ├── optimum_current.py             # Optimization algorithm
│   │   ├── rk4_error.py                   # Truncation error analysis
│   │   └── cooling_analysis.py            # Channel head-loss model
│   │
│   ├── analysis/                          # Parametric sensitivity studies
│   │   ├── __init__.py
│   │   ├── time_sensitivity.py            # RK4 step-size analysis
│   │   ├── current_sensitivity.py         # Current parameter sweep
│   │   └── massflow_sensitivity.py        # Mass flow parametric study
│   │
│   ├── utils/                             # Utility functions
│   │   ├── __init__.py
│   │   ├── heptane_interpolater.py        # Property correlations
│   │   ├── interpolater.py                # Spline routines
│   │   ├── root_finders.py                # Root-finding algorithms
│   │   ├── c_rate_interpolation.py        # C-rate calculations
│   │   └── clear_values.py                # Reset utilities
│   │
│   └── visualisation/                     # Plotting modules
│       ├── __init__.py
│       └── comparison_plot.py
│
├── scripts/                               # Executable entry points
│   ├── main.py                            # Primary analysis workflow
│   ├── gui.py                             # Interactive GUI application
│   └── charging_time_analysis.py          # Charging performance module
│
├── results/                               # Generated outputs (runtime)
│   ├── figures/                           # Visualization outputs
│   ├── tables/                            # Numerical results (CSV)
│   └── validation/                        # Sensitivity study results
│
└── key files/                             # Reference documentation
```

## 2. Installation and Environment Setup

### 2.1 System Requirements

- **Python:** 3.8 or higher (tested on 3.9, 3.10, 3.11, 3.13)
- **Git:** 
- **Storage:** ~500 MB for dependencies
- **OS:** Windows, macOS, Linux

### 2.2 Installation Procedure

```bash
# Clone repository
git clone <https://github.com/aksel010/Group-12-CMM3>
cd Group-12-CMM3

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install in editable mode
pip install -e .
```

### 3.3 Dependency Stack

| Package | Minimum Version | Purpose |
|---------|-----------------|---------|
| numpy | 1.24.0 | Numerical computing and linear algebra |
| scipy | 1.10.0 | Scientific algorithms and ODE integration |
| matplotlib | 3.7.0 | Publication-quality visualization |
| pandas | 2.0.0 | Data I/O and manipulation |
| seaborn | 0.12.0 | Statistical visualization |
| scikit-learn | 1.3.0 | Sensitivity analysis and regression |

## 4. Usage and Workflow

### 4.1 Command-Line Interface

Execute the complete analysis workflow:

```bash
python scripts/main.py
```

Output files are saved to `results/` directory with timestamped naming.

### 4.2 Graphical User Interface

Launch the interactive application:

```bash
python scripts/gui.py
```

### 5. Numerical Method Integration

## 6. Author Information

**Institution:** University of Edinburgh
**Department:** School of Engineering
**Completion Date:** November 2025
**Project Code:** CMM3 (Computational Modelling Module 3)
**Authors:** L. Curran, A. Huillard, G. Morgan, N. Tan

---

**Document Version:** 1.5.6
**Last Updated:** 19 November 2025
