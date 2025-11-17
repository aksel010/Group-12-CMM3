from config import *
import sys
import io

def calculate_charging_performance(critical_current, battery_capacity_Ah, efficiency=0.85):
    # Convert to C-rate and calculate times
    C_rate_critical = (critical_current) / Capacity_cell
    
    # Theoretical charging time
    theoretical_min = 60 / C_rate_critical
    
    # Practical charging time (with 80% current and 20% buffer)
    C_rate_practical = (critical_current * 0.8) / battery_capacity_Ah
    practical_min = (60 / C_rate_practical) * 1.2  # accounting for CC-CV charging
    
    return {
        'critical_C_rate': round(C_rate_critical, 2),
        'fastest_charge_min': round(theoretical_min, 1),
        'recommended_charge_min': round(practical_min, 1),
        'recommended_C_rate': round(C_rate_practical, 2)
    }

# Calculate and display results
def run():
    # Get critical current from Optimum_Current module
    print("Computing...")
    
    # Get critical current from I_store (already calculated in main)
    from config import I_store

    if len(I_store) == 0:
        print("Error: Optimum current not yet calculated!")
        return
    
    critical_current = I_store[-1]  # Get the converged critical current
    
    results = calculate_charging_performance(critical_current, Capacity_cell)

    print("\n--- Charging Performance Results ---")
    print(f"Optimum C-rate: {results['critical_C_rate']}C")
    print(f"Optimum charge time: {results['fastest_charge_min']} min")
    print(f"Recommended C-rate: {results['recommended_C_rate']} C ({results['recommended_charge_min']} min)")


if __name__ == "__main__":
    run()