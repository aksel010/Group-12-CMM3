from config import *
import sys
import io

# -------------------------------------------------------------------------
# FUNCTION TO CALCULATE CHARGING PERFORMANCE
# -------------------------------------------------------------------------
def calculate_charging_performance(critical_current, battery_capacity_Ah, efficiency=0.85):
    """
    Computes charging times and C-rates based on the critical current.

    Parameters:
        critical_current : float
            The maximum safe charging current [A]
        battery_capacity_Ah : float
            Battery capacity [Ah]
        efficiency : float, optional
            Charging efficiency (default: 0.85)

    Returns:
        dict with keys:
            - critical_C_rate : C-rate at critical current
            - fastest_charge_min : theoretical minimum charge time [min]
            - recommended_charge_min : practical charge time [min]
            - recommended_C_rate : practical C-rate
    """

    # Compute C-rate at the critical current
    C_rate_critical = critical_current / battery_capacity_Ah

    # Theoretical minimum charging time in minutes (assuming full current)
    theoretical_min = 60 / C_rate_critical

    # Practical charging: 80% of critical current, 20% buffer for CC-CV
    C_rate_practical = (critical_current * 0.8) / battery_capacity_Ah
    practical_min = (60 / C_rate_practical) * 1.2  # 20% buffer for safety

    return {
        'critical_C_rate': round(C_rate_critical, 2),
        'fastest_charge_min': round(theoretical_min, 1),
        'recommended_charge_min': round(practical_min, 1),
        'recommended_C_rate': round(C_rate_practical, 2)
    }


# -------------------------------------------------------------------------
# MAIN RUN FUNCTION
# -------------------------------------------------------------------------
def run():
    """
    Retrieves the critical current from I_store and computes
    charging performance metrics.
    """

    print("Computing...")

    # Retrieve the critical current calculated in previous module
    from config import I_store

    if len(I_store) == 0:
        print("Error: Optimum current not yet calculated!")
        return

    critical_current = I_store[-1]  # last stored critical current

    # Calculate charging performance
    results = calculate_charging_performance(critical_current, Capacity_cell)

    # Display results
    print("\n--- Charging Performance Results ---")
    print(f"Optimum C-rate: {results['critical_C_rate']} C")
    print(f"Optimum charge time: {results['fastest_charge_min']} min")
    print(f"Recommended C-rate: {results['recommended_C_rate']} C")
    print(f"Recommended charge time: {results['recommended_charge_min']} min")


# -------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run()
