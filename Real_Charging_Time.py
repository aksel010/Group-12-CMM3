from config import *
import sys
import io

def calculate_charging_performance(critical_current, battery_capacity_Ah, efficiency=0.85):
    """
    Computes charging times and C-rates based on the critical current.
    """
    c_rate_critical = critical_current / battery_capacity_Ah
    theoretical_min = 60 / c_rate_critical
    c_rate_practical = (critical_current * 0.8) / battery_capacity_Ah
    practical_min = (60 / c_rate_practical) * 1.2
    return {
        'critical_c_rate': round(c_rate_critical, 2),
        'fastest_charge_min': round(theoretical_min, 1),
        'recommended_charge_min': round(practical_min, 1),
        'recommended_c_rate': round(c_rate_practical, 2)
    }

def run():
    """
    Retrieves the critical current from I_store and computes
    charging performance metrics.
    """
    print("Computing...")
    from config import I_store
    if len(I_store) == 0:
        print("Error: Optimum current not yet calculated!")
        return
    critical_current = I_store[-1]
    results = calculate_charging_performance(critical_current, capacity_battery)
    print("\n--- Charging Performance Results ---")
    print(f"Optimum C-rate: {results['critical_c_rate']} C")
    print(f"Optimum charge time: {results['fastest_charge_min']} min")
    print(f"Recommended C-rate: {results['recommended_c_rate']} C")
    print(f"Recommended charge time: {results['recommended_charge_min']} min")

if __name__ == "__main__":
    run()
