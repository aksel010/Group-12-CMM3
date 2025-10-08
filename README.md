# Group-12-CMM3

# EV Battery Fast-Charging Cooling System Design

## Project Overview
This project optimizes the design of a liquid cooling plate for EV battery thermal management during fast charging. The goal is to minimize pump power consumption while maintaining safe battery temperatures.

### Team Members
- Aksel Huillard
- Leo Pullar
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

## Quick Start
```bash
# Clone repository
git clone https://github.com/aksel010/GROUP-12-CMM3.git

# Install dependencies
pip install -r requirements.txt

# Run main optimization
python src/main.py

# View results
python src/visualization.py