"""
Compute and report practical battery fast-charging rates and durations given an optimum current.
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

def monte_carlo_error_propagation(critical_current, delta_I, n_simulations=10000):
    """
    Monte Carlo simulation for error propagation.
    """
    # Define distributions for uncertain parameters
    current_samples = np.random.normal(critical_current, delta_I, n_simulations)
    capacity_samples = np.random.normal(6.0, 0.1*6.0, n_simulations)  # 6Ah ± 10%
    safety_samples = 0.8
    buffer_samples = 1.2
    
    # Run calculations for all samples
    C_rate_critical_samples = current_samples / capacity_samples
    theoretical_min_samples = 60 / C_rate_critical_samples
    
    C_rate_practical_samples = (current_samples * safety_samples) / capacity_samples
    practical_min_samples = (60 / C_rate_practical_samples) * buffer_samples
    
    # Compute statistics
    results = {
        'C_rate_critical': (np.mean(C_rate_critical_samples), np.std(C_rate_critical_samples)),
        'theoretical_min': (np.mean(theoretical_min_samples), np.std(theoretical_min_samples)),
        'C_rate_practical': (np.mean(C_rate_practical_samples), np.std(C_rate_practical_samples)),
        'practical_min': (np.mean(practical_min_samples), np.std(practical_min_samples))
    }
    
    print("Monte Carlo Results:")
    for key, (mean, std) in results.items():
        print(f"{key:20} = {mean:.3f} ± {std:.3f}")
    
    return results

def run():
    """
    Retrieve the latest critical current, compute and print practical and theoretical fast charge results.
    """
    print("Computing... (approx 40-120 seconds)")
    if not current_store:
        print("Error: Optimum current not yet calculated!")
        return None

    if not current_error:
        print("Error: Optimum current error not yet calculated!")
        return None
    critical_current = current_store[-1]
    results = calculate_charging_performance(critical_current, capacity_battery)
    monte_carlo_error_propagation(critical_current, current_error[-1])
    return results

if __name__ == "__main__":
    run()
