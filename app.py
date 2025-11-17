# app.py
import streamlit as st
import matplotlib.pyplot as plt
from model_runner import run_simulation

st.set_page_config(layout="wide", page_title="EV Battery Fast-Charge Optimizer")

st.title("EV Battery Fast-Charging Thermal Management System")
st.markdown("""
#### Academia-grade: Optimizes cooling plate design for HEVs during fast charge cycles.
*Simulation workflow, results visualization, and summary outputs available below.*
""")

# --- Config Panel ---
st.sidebar.header("Simulation Inputs")
I_0 = st.sidebar.number_input("Initial Current (A)", value=15)
n_cell = st.sidebar.number_input("Number of Cells", value=240)
T_in = st.sidebar.number_input("Coolant Entry Temperature (K)", value=288.13)

config = {"I_0": I_0, "n_cell": n_cell, "T_in": T_in}

if st.sidebar.button("Run Simulation"):
    results = run_simulation(config)

    # Results Display: Three-Panel Plots
    fig, axs = plt.subplots(1, 3, figsize=(18,6))
    # Panel 1: ODE Results
    axs[0].plot(*results['ode']['rk4'], label='RK4', linestyle='--', color='b')
    axs[0].plot(*results['ode']['scipy'], label='SciPy', color='r')
    axs[0].set_title("ODE Solution Comparison")
    axs[0].legend()
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Battery Temp (K)")

    # Panel 2: Pressure Balance
    axs[1].plot(results['mass_flow'], results['residuals'], color='purple')
    axs[1].set_title("Pressure Balance Residuals")
    axs[1].set_xlabel("Mass Flow Rate (kg/s)")
    axs[1].set_ylabel("Pressure Residual (Pa)")

    # Panel 3: Optimum Current
    axs[2].plot(results['optimum']['I_smooth'], results['optimum']['delta_T_smooth'], color='green')
    axs[2].scatter([results['optimum']['critical_current']], [0], color='red', s=100, label='Critical Point')
    axs[2].set_title("Optimum Current Analysis")
    axs[2].set_xlabel("Current (A)")
    axs[2].set_ylabel("ΔT (K)")
    axs[2].legend()

    st.pyplot(fig)

    # Show simulation log
    st.subheader("Computation Log")
    st.text_area("Log Output", results['log'], height=150)

    # Documentation (collapsible)
    with st.expander("Methodology, Assumptions, References"):
        st.markdown("""
        *Uses multi-objective optimization and mathematical modeling methods developed for CMM3 coursework in Mechanical Engineering, University of Edinburgh.*
        """)

    # Export button example
    st.download_button("Export ODE Results to CSV", "Time,Temp\n" + "\n".join([f"{t},{T}" for t, T in zip(results['ode']['rk4'][0], results['ode']['rk4'][1])]), "ode_solution.csv")

st.sidebar.info("Edit parameters and rerun for sensitivity/optimization studies.")

st.markdown("---")
st.markdown("Contributors: Aksel, Leo, Nicholas, Gregory | CMM3 University of Edinburgh | November 2025")
