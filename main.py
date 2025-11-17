# main.py — Group-12-CMM3 Consolidated Execution
import numpy as np
import matplotlib.pyplot as plt

# Import configuration and all modules' run functions
from config import *
import ODE
import RK4_Error as rk4e
import Mass_flowrate as mf
import Optimum_Current as oc
import Real_Charging_Time as rct
import cooling_analysis as ca
import heptane_itpl as hi

def compute_optimum_current(threshold=I_Threshold):
    I_store.clear()  # Start fresh for each run
    print("\n--- Optimum Current Analysis ---")
    current = oc.run()  # first value
    I_store.append(current)

    while True:
        new_current = oc.run()
        I_store.append(new_current)
        # Only check convergence after there are at least 2 values
        if len(I_store) > 1 and abs(I_store[-1] - I_store[-2]) < threshold:
            break
    return I_store

def main():
    print("==== Group 12 - CMM3 Consolidated Results ====")
    
    ODE.run()

    print("\n--- RK4 Error Analysis ---")
    rk4e.run()

    print("\n--- Mass Flowrate Solver ---")
    mf.run()

    print("\n--- Real Charging Time ---")
    rct.run()

    hi.run()

if __name__ == "__main__":
    main()
    I_store = compute_optimum_current(threshold=1e-6)
    print(f'Model iterated {len(I_store)} times')
    print(I_store)
    print("\n✓ All computations and plots complete!")
    
    plt.show()
    
