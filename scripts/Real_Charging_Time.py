"""
Compute and report practical battery fast-charging rates and durations given an optimum current, with clear docstrings and comments.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import *

def calculate_charging_performance(critical_current, battery_capacity_Ah, efficiency=0.85):
    """
    Calculates practical and theoretical charging times and C-rates given a critical current.

    Args:
        critical_current (float): The maximum allowable charging current [A].
        battery_capacity_Ah (float): Battery capacity [Ah].
        efficiency (float, optional): Charging efficiency (default 0.85, but not used in logic).
    Returns:
        dict: {'critical_C_rate', 'fastest_charge_min', 'recommended_charge_min', 'recommended_C_rate'}
    """
    # Theoretical C-rate and minimum charge time
    C_rate_critical = critical_current / battery_capacity_Ah
    theoretical_min = 60 / C_rate_critical
    # Practical C-rate: buffer for safety and account for CC/CV segment
    C_rate_practical = (critical_current * 0.8) / battery_capacity_Ah
    practical_min = (60 / C_rate_practical) * 1.2
    return {
        'critical_C_rate': round(C_rate_critical, 2),
        'fastest_charge_min': round(theoretical_min, 1),
        'recommended_charge_min': round(practical_min, 1),
        'recommended_C_rate': round(C_rate_practical, 2)
    }

def run():
    """
    Retrieve the latest critical current, compute and print practical and theoretical fast charge results.
    """
    print("Computing...")
    if len(current_store) == 1:
        print("Error: Optimum current not yet calculated!")
        return
    critical_current = current_store[-1]
    results = calculate_charging_performance(critical_current, capacity_battery)
    print("\n--- Charging Performance Results ---")
    print(f"Optimum C-rate: {results['critical_C_rate']} C")
    print(f"Optimum charge time: {results['fastest_charge_min']} min")
    print(f"Recommended C-rate: {results['recommended_C_rate']} C")
    print(f"Recommended charge time: {results['recommended_charge_min']} min")

if __name__ == "__main__":
    run()
