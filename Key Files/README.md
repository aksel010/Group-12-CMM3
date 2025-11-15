# Group-12-CMM3

## EV Battery Fast-Charging Thermal Management System

### Project Overview

This project designs and optimises a liquid cooling plate system for a hybrid electric vehicle (HEV) battery thermal management during fast-charging cycles. The simulation framework evaluates cooling channel geometries and flow conditions to minimize pump power consumption while maintaining safe battery operating temperatures.

**Key Challenge**: Fast charging (3C rate) generates significant heat in lithium-ion battery cells. Without effective cooling, cell temperatures can exceed safe operating limits (>45°C), reducing battery lifespan and creating safety risks.

**Solution Approach**: Multi-objective optimization of cooling plate design parameters (channel dimensions, number of channels, flow rate) to balance thermal performance against pumping power requirements.

---

### Team Members
- **Aksel Huillard**
- **Leo Curran**
- **Nicholas Tan**
- **Gregory Morgan**

**Course**: CMM3 - Computational Methods and Modelling  
**Program**: 3rd Year Mechanical Engineering  
**Institution**: The University of Edinburgh

---

## Design Objectives

The optimization targets three critical constraints:

1. **Thermal Constraint**: Maximum cell temperature ≤ 45°C during 3C fast charge
2. **Pressure Constraint**: Total pressure drop ≤ 100 kPa across cooling system
3. **Optimization Goal**: Minimize pumping power time

**Design Variables**:
- Channel width and height
- Number of parallel cooling channels
- Coolant flow rate (n-heptane)

---

## Methodology

The simulation framework follows a systematic computational workflow:

### 1. Problem Definition
- Define battery cell geometry and material properties
- Specify cooling plate configuration and constraints
- Set design variable ranges for optimization space

### 2. Thermal Modeling
- Simulate transient heat generation during fast charging
- Model heat transfer from cells to cooling plate
- Calculate temperature distribution across battery module
- Uses ODE solvers (RK4 method) for time-dependent thermal response

### 3. Fluid Dynamics Analysis
- Calculate Reynolds number and flow regime
- Determine friction factors for pressure drop calculations
- Evaluate heat transfer coefficients (Nusselt correlations)
- Compute required pumping power for each design configuration

### 4. Multi-Objective Optimization
- Systematic exploration of design space
- Constraint satisfaction checking (temperature and pressure limits)
- Identification of Pareto-optimal solutions
- Trade-off analysis between thermal performance and power consumption

### 5. Results Visualization
- Generate performance plots and design charts
- Visualize optimal design parameters
- Compare cooling effectiveness across configurations

---

## Repository Structure

### Core Files

#### `config.py`
**Purpose**: Centralized configuration management  
**Contains**:
- Physical constants (thermal conductivity, specific heat, density)
- Battery cell properties (dimensions, internal resistance, capacity)
- Coolant properties (n-heptane fluid characteristics)
- Geometric parameters (channel dimensions, module layout)
- Simulation settings (time steps, convergence criteria)

**Usage**: Import this module in all other scripts to ensure consistent parameter values

---

#### `main.py`
**Purpose**: Main optimization driver script  
**Function**:
- Coordinates the entire optimization workflow
- Generates design space (combinations of channel dimensions and flow rates)
- Calls thermal and fluid dynamics models for each configuration
- Evaluates objective function (pumping power) and constraints
- Identifies optimal design from simulation results
- Exports optimization results to file

**Inputs**: Configuration from `config.py`  
**Outputs**: 
- Optimal design parameters
- Performance metrics (max temperature, pressure drop, pump power)
- Raw optimization data for post-processing

---

#### `clear_val.py`
**Purpose**: Battery thermal modeling  
**Function**:
- Calculates heat generation rate based on cell current and internal resistance
- Models transient thermal response of battery cells
- Simulates temperature evolution during fast-charging cycle
- Accounts for heat transfer to cooling plate

**Inputs**: 
- C-rate (charging current multiplier)
- Initial cell temperature
- Cooling conditions (flow rate, channel geometry)

**Outputs**: 
- Cell temperature profile over time
- Maximum cell temperature reached
- Heat flux to cooling system

---

#### `cooling_analysis.py`
**Purpose**: Cooling system fluid dynamics  
**Function**:
- Calculates Reynolds number for flow characterization
- Determines flow regime (laminar/turbulent)
- Computes friction factor using appropriate correlations
- Evaluates pressure drop across cooling channels
- Calculates heat transfer coefficients

**Inputs**: 
- Channel dimensions (width, height, length)
- Flow rate and fluid properties
- Number of parallel channels

**Outputs**: 
- Pressure drop (Pa)
- Pumping power (W)
- Heat transfer coefficient (W/m²K)
- Reynolds and Nusselt numbers

---

#### `ODE.py`
**Purpose**: Ordinary differential equation solvers  
**Function**:
- Implements 4th-order Runge-Kutta (RK4) method
- Solves time-dependent thermal equations
- Provides numerical integration for temperature evolution

**Inputs**: 
- Differential equation system
- Initial conditions
- Time step and integration period

**Outputs**: 
- Solution trajectory over time
- Temperature profiles

---

#### `heptane_itpl.py`
**Purpose**: N-heptane fluid property interpolation  
**Function**:
- Interpolates thermophysical properties of n-heptane coolant
- Properties vary with temperature (density, viscosity, thermal conductivity)
- Uses lookup tables and interpolation for accurate property values

**Data Source**: `n heptane 2.txt` (property database)  
**Outputs**: Temperature-dependent fluid properties for cooling calculations

---

#### `Mass_flowrate.py`
**Purpose**: Flow distribution calculations  
**Function**:
- Calculates mass flow rate distribution across parallel channels
- Converts volumetric flow rate to mass flow rate
- Accounts for flow splitting and channel resistance

**Inputs**: Total flow rate, number of channels, fluid density  
**Outputs**: Per-channel mass flow rates

---

#### `Current_Error.py`
**Purpose**: Error analysis and validation  
**Function**:
- Performs sensitivity analysis on key parameters
- Evaluates numerical error in flow calculations
- Validates model predictions against analytical solutions

**Outputs**: Error metrics and convergence analysis

---

#### `Optimum_Current.py`
**Purpose**: Charging current optimization  
**Function**:
- Determines optimal charging current profile
- Balances charging speed against thermal constraints
- Analyzes trade-offs between C-rate and battery temperature

**Outputs**: Recommended charging strategies for different cooling configurations

---

#### `Real_Charging_Time.py`
**Purpose**: Realistic charging time estimation  
**Function**:
- Calculates actual charging time considering thermal limits
- Accounts for current throttling due to temperature constraints
- Simulates real-world fast-charging scenarios

**Outputs**: 
- Charging time estimates
- Temperature-limited charging curves

---

#### `C-Rate_Real_interpolation.py`
**Purpose**: Results visualization and analysis  
**Function**:
- Generates performance plots from optimization results
- Creates design charts (e.g., temperature vs. flow rate)
- Visualizes trade-offs between competing objectives
- Interpolates C-rate data for smooth plotting

**Data Source**: `Real_data_c-rate.txt` (experimental/validation data)  
**Outputs**: Figures and plots for report and presentation

---

### Data Files

- **`Real_data_c-rate.txt`**: Experimental or reference data for C-rate validation
- **`n heptane 2.txt`**: Thermophysical property database for n-heptane coolant
- **`requirements.txt`**: Python package dependencies
- **`.gitignore`**: Git version control ignore rules

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/aksel010/Group-12-CMM3.git
cd Group-12-CMM3/Key\ Files/

# Install required Python packages
pip install -r requirements.txt
```

### Required Python Packages
- `numpy` - Numerical computing
- `scipy` - Scientific computing and optimization
- `matplotlib` - Data visualization
- `pandas` (if used) - Data manipulation

All dependencies are listed in `requirements.txt`.

---

## Usage

### Running the Optimization

```bash
# Navigate to Key Files directory
cd "Key Files"

# Run main optimization
python main.py
```

The optimization will:
1. Load configuration from `config.py`
2. Generate design space
3. Evaluate each design configuration
4. Output optimal design parameters
5. Save results to file

### Viewing Results

```bash
# Generate visualization plots
python C-Rate_Real_interpolation.py
```

This creates performance plots showing:
- Temperature vs. flow rate
- Pumping power vs. channel geometry
- Pareto frontiers for multi-objective trade-offs

### Customizing Parameters

To modify simulation parameters:
1. Edit `config.py` to change battery properties, constraints, or design ranges
2. Re-run `main.py` to evaluate new configurations

---

## Key Results

The optimization identifies cooling plate designs that:
- **Maintain thermal safety**: Cell temperatures stay below 45°C
- **Minimize energy consumption**: Reduced pumping power compared to baseline designs
- **Meet pressure constraints**: Pressure drop within system limits (≤100 kPa)

Typical optimal configurations feature:
- Serpentine or parallel channel layouts
- Balanced trade-off between channel size and number
- Flow rates optimized for heat removal vs. pumping power

---

## Technical Approach

### Heat Generation Model
- Heat generation rate: `Q = I²R` (Joule heating from internal resistance)
- 3C charging: Current = 3× cell capacity
- Time-dependent temperature rise calculated using thermal mass and heat transfer

### Cooling System Design
- **Coolant**: N-heptane (selected for thermal properties and safety)
- **Channel geometry**: Rectangular cross-section, optimized dimensions
- **Flow configuration**: Parallel channels with common inlet/outlet manifolds

### Pressure Drop Calculation
- Darcy-Weisbach equation: `ΔP = f × (L/D) × (ρv²/2)`
- Friction factor correlations for laminar and turbulent flow
- Minor losses from inlet/outlet effects

### Pumping Power
- `P_pump = ṁ × ΔP / (ρ × η_pump)`
- Assumed pump efficiency: 70-80%

---

## Limitations and Assumptions

1. **Simplified geometry**: 2D thermal model, uniform heat generation
2. **Steady coolant properties**: Temperature-dependent properties interpolated but flow assumed incompressible
3. **Uniform cooling**: Equal flow distribution assumed across parallel channels
4. **No phase change**: Single-phase liquid cooling only
5. **Ideal contact**: Perfect thermal contact between cells and cooling plate

Future work could address these through:
- 3D CFD simulations
- Conjugate heat transfer analysis
- Flow distribution optimization
- Two-phase cooling strategies

---

## References and Background

### Battery Thermal Management
- Fast charging generates heat that must be removed to prevent degradation
- Optimal operating temperature range: 25-40°C
- Exceeding 45°C accelerates capacity fade and increases safety risks

### Cooling Strategies
1. **Air cooling**: Simple but limited heat removal capacity
2. **Liquid cooling**: High heat transfer, used in this project
3. **Phase change cooling**: Emerging technology for extreme fast charging

### Application Context
- EV fast charging (DC fast charge, >50 kW)
- High-performance battery packs (Tesla, Porsche, etc.)
- Cylindrical 21700 or 18650 cell formats

---

## Contributing

This is an academic project for CMM3 coursework. For questions or collaboration:
- Open an issue on GitHub
- Contact team members through Imperial College

---

## License

Academic project - Imperial College London  
For educational and research purposes

---

## Acknowledgments

- **Course**: CMM3 Computational Methods and Modelling
- **Department**: Mechanical Engineering, Imperial College London
- **Instructors**: CMM3 teaching team
- **Resources**: Imperial College HPC facilities (if used)

---

## Project Status

🟢 **Active Development** - Optimization and analysis ongoing

Last Updated: November 2025
