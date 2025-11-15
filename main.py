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

def main():
    print("==== Group 12 - CMM3 Consolidated Results ====\n")
    
    print("\n--- BTMS (ODE) Optimization ---")
    ODE.run()

    print("\n--- RK4 Error Analysis ---")
    rk4e.run()

    print("\n--- Mass Flowrate Solver ---")
    mf.run()

    print("\n--- Optimum Current Analysis ---")
    oc.run()

    print("\n--- Real Charging Time ---")
    rct.run()

    print("\n--- Cooling Analysis ---")
    ca.run()

    print("\n--- Heptane Fluid Properties ---")
    hi.run()
    
    print("\n✓ All computations and plots complete!")

if __name__ == "__main__":
    main()
