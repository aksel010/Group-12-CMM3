import numpy as np
import matplotlib.pyplot as plt
from ODE import *
from config import *
from Current_against_delta_T import *

def comprehensive_validation(I_array, delta_T_array, critical_current, 
                           battery_capacity_Ah, T_b_max):
    """
    Complete validation of your charging optimization
    """
    
    print("=== CHARGING OPTIMIZATION VALIDATION ===")
    
    # 1. Interpolation Quality Check
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(I_array, delta_T_array, color='blue', s=50, label='Data')
    I_smooth = np.linspace(I_array.min(), I_array.max(), 100)
    delta_T_smooth = [cubic_spline_interpolation(I_array, delta_T_array, I) for I in I_smooth]
    plt.plot(I_smooth, delta_T_smooth, 'r-', label='Spline Fit')
    plt.axhline(0, color='black', linestyle='--', label='Safety Limit')
    plt.axvline(critical_current, color='green', linestyle=':', label='Critical Current')
    plt.xlabel('Current (A)'); plt.ylabel('ΔT (K)')
    plt.title('Model Validation'); plt.legend()
    
    # 2. Safety Analysis
    safety = calculate_safety_factors(critical_current, critical_current*0.9, 
                                    delta_T_array + T_b_max, T_b_max)
    
    # 3. Charging Performance
    charging = calculate_charging_performance(critical_current, battery_capacity_Ah)
    
    # Print results
    print(f"\n--- SAFETY ANALYSIS ---")
    print(f"Critical Current: {critical_current:.2f} A")
    print(f"Thermal Safety Margin: {safety['thermal_safety_margin_K']:.2f} K")
    print(f"Recommended Operating Current: {safety['recommended_operating_current_A']:.2f} A")
    
    print(f"\n--- CHARGING PERFORMANCE ---")
    print(f"Critical C-rate: {charging['critical_C_rate']:.2f}C")
    print(f"Theoretical Fastest Charge: {charging['theoretical_fastest_charge_min']:.1f} min")
    print(f"Recommended Charge Time: {charging['recommended_charge_time_min']:.1f} min")
    
    
    plt.subplot(1, 2, 2)
    rates = list(charging['industry_comparison'].values())
    names = list(charging['industry_comparison'].keys())
    plt.barh(names, rates, color='lightblue', label='Industry Standards')
    plt.axvline(charging['recommended_C_rate'], color='red', 
                linestyle='--', label='Your System')
    plt.xlabel('C-rate'); plt.title('Performance Benchmarking')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return safety, charging

# Run the validation
battery_capacity = 5.0  # Ah - you'll need to set this based on your actual battery
safety_results, charging_results = comprehensive_validation(
    I_array, delta_T_array, critical_current, battery_capacity, T_b_max
)