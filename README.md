# Advanced Battery Thermal Management System (BTMS) for PHEVs Using Fuel-Based Cooling

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## Overview

This repository contains a Python-based computational framework for simulating and optimizing an innovative Battery Thermal Management System (BTMS) for Plug-in Hybrid Electric Vehicles (PHEVs). The system employs the vehicle's fuel (n-heptane) as coolant medium, offering superior thermal performance compared to conventional water-glycol systems.

**Research Objectives:**
- Determine minimum feasible charging time while maintaining battery temperature below 40°C
- Maximize charging current within thermal constraints

## Key Features

- Multi-physics coupling: thermal, fluid dynamics, and electrochemical models
- Transient analysis with multi-node spatial discretization
- Automated optimization using parallel processing
- Interactive GUI and command-line interfaces
- NIST-validated thermophysical property correlations

## Project Structure

```
btms-simulation/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Version control exclusions
│
├── data/                          # Thermophysical data
│   ├── raw/                       # Original NIST data
│   └── processed/                 # Fitted polynomial coefficients
│
├── src/                           # Source code
│   ├── models/                    # Core physics models
│   │   ├── battery_model.py       # Thermal and electrochemical
│   │   ├── coolant_model.py       # n-heptane properties
│   │   ├── flow_model.py          # Fluid dynamics
│   │   └── geometry_model.py      # Channel configurations
│   │
│   ├── analysis/                  # Sensitivity studies
│   │   └── optimization_engine.py # Multi-objective optimization
│   │
│   ├── utils/                     # Helper functions
│   │   ├── simulation_runner.py   # Orchestration
│   │   ├── visualization.py       # Plotting
│   │   └── process_nist_data.py   # Data preprocessing
│   │
│   ├── config.py                  # Configuration parameters
│   └── gui_app.py                 # Graphical interface
│
├── scripts/                       # Executable entry points
│   └── main_simulation.py         # CLI interface
│
├── tests/                         # Unit tests and validation
│   ├── test_battery_model.py
│   ├── test_coolant_model.py
│   └── test_flow_model.py
│
├── docs/                          # Documentation
│   ├── methodology.md             # Mathematical formulation
│   └── user_guide.md              # Usage instructions
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
git clone https://github.com/GPMorg/Greg-Git.git
cd Greg-Git

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Process thermophysical data (first-time only)
python src/utils/process_nist_data.py
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
- Channel layout diagrams

### Command-Line Interface

```bash
python scripts/main_simulation.py
```

Outputs optimal design parameters:
- Number of cooling channels
- Charging current (A)
- Minimum charging time (s)
- Maximum battery temperature (°C)

### Python API

```python
from src.models.battery_model import BatteryPack
from src.models.coolant_model import CoolantSystem
from src.models.flow_model import FlowSystem
from src.models.geometry_model import GeometryConfig
from src.analysis.optimization_engine import Optimizer

# Initialize models
battery = BatteryPack(mass=10.0, cp=1000.0, resistance=0.05, 
                      capacity_ah=27.0, max_temp_c=35.0)
coolant = CoolantSystem()
geometry = GeometryConfig(channel_length=0.5, channel_width=0.002, 
                          channel_height=0.001)
flow = FlowSystem(max_pressure_pa=50000.0, coolant_system=coolant)

# Configure simulation
config = {
    'battery_max_temp': 35.0,
    'coolant_inlet_temp': 20.0,
    'time_end': 3600.0,
    'time_step': 1.0,
    'initial_temp': 25.0,
    'layout_type': 'Parallel'
}

# Run optimization
optimizer = Optimizer(battery, coolant, flow, geometry, config)
I_max = optimizer.find_max_current(N_channels=10)
results = optimizer.run_simulation(N=10, current=I_max)

print(f"Maximum temperature: {results['T_b'].max() - 273.15:.2f}°C")
```

## Mathematical Framework

### Thermal Dynamics

Battery node energy balance:

$$\frac{dT_{b,i}}{dt} = \frac{Q_{gen,i} - Q_{cool,i} + Q_{cond,i}}{m_i c_{p,i}}$$

Where:
- $Q_{gen} = I^2 R$ (Joule heating)
- $Q_{cool}$ = convective heat removal
- $Q_{cond}$ = inter-node conduction

### Heat Transfer

Effectiveness-NTU method:

$$NTU = \frac{hA}{\dot{m} c_p}, \quad \epsilon = 1 - e^{-NTU}$$

$$Q_{cool} = \epsilon \dot{m} c_p (T_b - T_{c,in})$$

Nusselt number (Dittus-Boelter correlation):

$$Nu = 0.023 Re^{0.8} Pr^{0.4}, \quad h = \frac{Nu \cdot k}{D_h}$$

### Fluid Dynamics

Darcy-Weisbach equation for pressure drop:

$$\Delta P = f \frac{L}{D_h} \frac{\rho u^2}{2} + K_{minor} \frac{\rho u^2}{2}$$

Reynolds number:

$$Re = \frac{\rho u D_h}{\mu}$$

Friction factor (Blasius, turbulent):

$$f = \frac{0.316}{Re^{0.25}} \quad (Re > 2300)$$

## Configuration Parameters

### Battery Specifications

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| Mass | 10.0 | kg | Battery pack mass |
| Specific heat | 1000.0 | J/(kg·K) | Heat capacity |
| Resistance | 0.05 | Ω | Internal resistance |
| Capacity | 27.0 | Ah | Charge capacity |
| Max temperature | 35.0 | °C | Thermal limit |

### Channel Geometry

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| Length | 0.5 | m | Channel length |
| Width | 0.002 | m | Channel width |
| Height | 0.001 | m | Channel height |

### Operating Conditions

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| Inlet temperature | 20.0 | °C | Coolant inlet |
| Max pressure | 50000.0 | Pa | Pump limit |
| Simulation time | 7200.0 | s | Duration |

## Validation and Verification

### Code Verification

1. **Mesh independence study**: Temperature convergence with node count (N = 5, 10, 20, 40)
2. **Time-step convergence**: Solution accuracy with varying time steps
3. **Energy balance**: Verification that $\int Q_{gen} dt = \int Q_{cool} dt + \Delta E_{stored}$

### Physical Validation

1. **Literature comparison**: Results validated against Al Qubeissi et al. (2022)
2. **Thermodynamic consistency**: Second law verification (entropy generation > 0)
3. **Limiting cases**: Zero current, infinite flow rate, adiabatic conditions

## Results and Outputs

### Console Output

```
--- Optimization Complete ---
  Optimal Number of Channels: 12 (Parallel Layout)
  Optimal Charging Current: 145.32 A
  Minimum Charging Time: 668.54 s (0.19 hrs)
  Predicted Max Battery Temp: 34.98 °C
```

### Visualization

- Temperature evolution plots (time vs. temperature)
- Channel layout diagrams (saved as PNG)
- Sensitivity analysis charts

### Data Export

- CSV: Time-series temperature data
- JSON: Complete simulation results
- PNG: Publication-quality figures (300 DPI)

## Troubleshooting

### Common Issues

**Issue**: "Fit coefficient file not found"
```bash
python src/utils/process_nist_data.py
```

**Issue**: ODE solver convergence failure
- Reduce time step in configuration
- Use BDF method for stiff systems: `method='BDF'`
- Increase tolerances: `rtol=1e-5, atol=1e-7`

**Issue**: GUI not displaying on remote server
```bash
# Enable X11 forwarding
ssh -X user@server
export DISPLAY=:0
```

## Development Team

**University of Edinburgh**

- Nicholas Tan - Thermal modeling and validation
- Gregory Morgan - Software architecture and optimization
- Aksel Huillard - Fluid dynamics and heat transfer
- Leo Pullar - Data processing and visualization

**Contact**: s2297115@sms.ed.ac.uk

## References

1. Al Qubeissi, M., Almshahy, A., et al. "Modelling of battery thermal management: A new concept of cooling using fuel." *Fuel*, vol. 310, 2022.

2. NIST Chemistry WebBook. "n-Heptane Thermophysical Properties." https://webbook.nist.gov/cgi/cbook.cgi?ID=C142825

3. Incropera, F. P., & DeWitt, D. P. *Fundamentals of Heat and Mass Transfer* (6th ed.). John Wiley & Sons, 2007.

4. White, F. M. *Fluid Mechanics* (7th ed.). McGraw-Hill, 2011.

## Citation

### BibTeX

```bibtex
@software{btms_phev_2025,
  title = {Advanced Battery Thermal Management System for PHEVs Using Fuel-Based Cooling},
  author = {Tan, Nicholas and Morgan, Gregory and Huillard, Aksel and Pullar, Leo},
  institution = {University of Edinburgh},
  year = {2025},
  url = {https://github.com/GPMorg/Greg-Git},
  version = {1.0.0}
}
```

### APA

Tan, N., Morgan, G., Huillard, A., & Pullar, L. (2025). *Advanced Battery Thermal Management System for PHEVs Using Fuel-Based Cooling* (Version 1.0.0) [Computer software]. University of Edinburgh. https://github.com/GPMorg/Greg-Git

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.**

---

**Built at the University of Edinburgh | 2025**