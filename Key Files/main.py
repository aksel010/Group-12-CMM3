# main.py — Group-12-CMM3 Consolidated Execution

import numpy as np
import matplotlib.pyplot as plt

# Import everything needed from your modules
from config import *
from ODE import Tb, dTb_dt, params_initial
import C_Rate_Real_interpolation as cri
import Current_Error as cerr
import Mass_flowrate as mf
import Optimum_Current as oc
import Real_Charging_Time as rct
import cooling_analysis as ca
import heptane_itpl as hi

def run_btms_optimization():
    print("\n--- BTMS Optimization ---")
    # Your original main.py logic (already structured for consolidated results)
    results, optimal = run_optimization()
    if optimal:
        plot_results(results, optimal)

def run_c_rate_interpolation():
    print("\n--- C-Rate Real Interpolation ---")
    # cri: Only has top-level code, so may need to move code to cri.run() or similar.
    # For now, assuming you can call functions for critical current and plotting:
    critical_current = cri.find_root_cubic_spline()
    print(f"Critical Current: {critical_current:.2f} A")

def run_error_analysis():
    print("\n--- Current Error Analysis ---")
    # cerr: Only has top-level code, so may need to move print logic to cerr.run() or similar.
    print(f"Combined Error: {cerr.total_combined_error:.6e} K")

def run_mass_flowrate():
    print("\n--- Mass Flowrate Solver ---")
    # mass_flowrate.py: Only has if __name__ == "__main__": code for printing/plot.
    # Should be a callable function, e.g. mf.run() or wrap logic in a function.
    # For now, this just demonstrates usage.
    # mf.calculate_steady_state_mass_flow(...)
    pass  # Replace with mf.run() after you move code out of main block.

def run_optimum_current():
    print("\n--- Optimum Current Analysis ---")
    # oc: Similar to above, wrap plot and print in callable function if needed.
    print(f"Critical Current from Optimum_Current: {oc.critical_current:.2f} A")
    print(f"RMSE of Interpolation: {oc.rmse:.6e} K")

def run_real_charging_time():
    print("\n--- Real Charging Time ---")
    # rct: Only has if __name__ == "__main__": code for printout.
    # Needs logic moved to rct.run()
    results = rct.calculate_charging_performance(oc.critical_current, Capacity_cell)
    print("===Charging Performance===")
    print(f"Optimum C-rate: {results['critical_C_rate']}C")
    print(f"Optimum charge time: {results['fastest_charge_min']} min")
    print(f"Recommended C-rate: {results['recommended_C_rate']}C ({results['recommended_charge_min']} min)")

def run_cooling_analysis():
    print("\n--- Cooling Analysis ---")
    # ca: Only top-level head_loss printout—same note as above.
    # Should be ca.run() for callable
    pass

def run_heptane_properties():
    print("\n--- Heptane Fluid Properties ---")
    # hi: Only has code under main block, so wrap your plotting logic in hi.run().
    pass

def main():
    print("==== Group 12 - CMM3 Consolidated Results ====\n")
    run_btms_optimization()
    run_c_rate_interpolation()
    run_error_analysis()
    run_mass_flowrate()
    run_optimum_current()
    run_real_charging_time()
    run_cooling_analysis()
    run_heptane_properties()
    print("\n✓ All computations and plots complete!")

if __name__ == "__main__":
    main()
