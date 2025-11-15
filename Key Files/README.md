# Group-12-CMM3

# EV Battery Fast-Charging Cooling System Design

## Project Overview
This project optimizes the design of a liquid cooling plate for EV battery thermal management during fast charging. The goal is to minimize pump power consumption while maintaining safe battery temperatures.

### Team Members
- Aksel Huillard
- Leo Curran
- Nicholas Tan
- Gregory Morgan

### Course
CMM3 - Computational Methods and Modelling
3rd Year Mechanical Engineering

## Project Objectives
Design cooling channel geometry (width, height, number of channels) and flow rate to:
- Keep maximum cell temperature below 45°C during 3C fast charge
- Minimize pumping power consumption
- Meet pressure drop constraints (ΔP ≤ 100 kPa)

## Project Workflow
The project follows these computational steps to find the optimal cooling plate design:

1.  **Problem Definition**: The physical and geometric parameters of the battery cell and cooling plate are defined. This includes material properties, heat generation rates, and the range of design variables to be explored.
2.  **Thermal Modeling**: A thermal model is executed to simulate the temperature distribution across the battery cells during a 3C fast-charging cycle.
3.  **Fluid Dynamics**: For each cooling channel geometry and flow rate, the pressure drop and required pumping power are calculated.
4.  **Optimization**: An optimization algorithm iterates through the design space. It uses the outputs from the thermal and fluid dynamics models to find a set of design parameters (channel dimensions, flow rate) that minimizes pumping power while keeping the maximum battery temperature below 45°C and the pressure drop under 100 kPa.
5.  **Data Visualization**: The results from the optimization are processed and plotted to visualize the optimal design and explore the trade-offs between different parameters.

## Code Structure

### `config.py`
This file centralizes all the physical and geometric constants used across the project.
*   **Inputs**: None.
*   **Outputs**: Provides configuration variables (e.g., battery properties, fluid properties, geometric dimensions) that are imported by other modules.

### `main.py`
This is the main script that drives the optimization process.


### `C-Rate_Real_interpolation`
This script is used to generate plots and figures from the optimization results.
*   **Inputs**:
*   **Outputs**:

### `clear_val.py`
Contains the functions and logic for the battery thermal model.
*   **Inputs**:
*   **Outputs**:

### `cooling_analysis.py`
Contains the functions for calculating fluid dynamic properties.
*   **Inputs**:
*   **Outputs**:Files/config.py`.

### `Current_Error.py`
Contains the functions for calculating fluid dynamic properties.
*   **Inputs**:
*   **Outputs**:Files/config.py`.

### `heptane_itpl.py`
Contains the functions for calculating fluid dynamic properties.
*   **Inputs**:
*   **Outputs**:Files/config.py`.

### `Mass_flowrate.py`
Contains the functions for calculating fluid dynamic properties.
*   **Inputs**:
*   **Outputs**:Files/config.py`.

### `ODE.py`
Contains the functions for calculating fluid dynamic properties.
*   **Inputs**:
*   **Outputs**:Files/config.py`.

### `Optimum_Current.py`
Contains the functions for calculating fluid dynamic properties.
*   **Inputs**:
*   **Outputs**:Files/config.py`.

### `requirements.txt`
A text file listing all the Python libraries and dependencies required to run the project.
*   **Inputs**: None.
*   **Outputs**: Used by the `pip` package manager to install the necessary project dependencies.

## Quick Start
```bash
# Clone repository
git clone https://github.com/aksel010/GROUP-12-CMM3.git

# Install dependencies
pip install -r requirements.txt

# Run main optimization
python "Key Files/main.py"

# View results
python "Key Files/visualization.py"
