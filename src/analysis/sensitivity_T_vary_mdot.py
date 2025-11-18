import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from config import *
from src.utils.heptane_itpl import lambda_func, Cp_func, mu_func, rho_func
from src.models.ODE import dTb_dt

# -------------------------------------------------------------------
# Function: calculate_h
# -------------------------------------------------------------------
def calculate_h(T: float | np.ndarray, m_dot: float) -> float | np.ndarray:
    """
    Calculate the convective heat transfer coefficient (h) based on
    coolant temperature and mass flow rate using Dittus-Boelter correlation.

    Parameters
    ----------
    T : float | np.ndarray
        Coolant temperature (K)
    m_dot : float
        Total mass flow rate (kg/s)

    Returns
    -------
    h : float | np.ndarray
        Convective heat transfer coefficient (W/m²K)
    """
    # Mass flow coefficient for circular equivalent
    C_RE_for_h = 4 * m_dot / (np.pi * D)

    # Fluid properties
    lam = lambda_func(T)
    Cp = Cp_func(T)
    mu = mu_func(T)

    # Reynolds and Prandtl numbers
    Re = C_RE_for_h / mu
    Pr = Cp * mu / lam

    # Nusselt number (Dittus-Boelter)
    Nu = 0.023 * (Re ** 0.8) * (Pr ** DITTUS_BOELTER_EXPONENT)

    # Heat transfer coefficient
    h = (Nu * lam) / D
    return h


# -------------------------------------------------------------------
# Function: dTb_dt_sensitivity
# -------------------------------------------------------------------
def dTb_dt_sensitivity(Tb, t, params):
    """
    Differential equation for battery bulk temperature Tb for sensitivity
    analysis with fixed mass flow rate.

    Parameters
    ----------
    Tb : float
        Bulk battery temperature (K)
    t : float
        Time (s)
    params : tuple
        (m, cp_b, I, R, A_s, T_c_in, m_dot_c)

    Returns
    -------
    dTb_dt : float
        Rate of change of bulk temperature (K/s)
    """
    m, cp_b, I, R, A_s, T_c_in, m_dot_c = params

    # Average coolant temperature
    T_c_avg = (Tb + T_c_in) / 2
    h = calculate_h(T_c_avg, m_dot_c)
    cp_c = Cp_func(T_c_avg)

    # Heating and cooling terms
    heating = I ** 2 * R
    cooling_denom = 1 + (h * A_s) / (2 * (m_dot_c / 4) * cp_c)
    cooling = (h * A_s * (Tb - T_c_in)) / cooling_denom

    return (heating - cooling) / (m * cp_b)


# -------------------------------------------------------------------
# Function: solve_ode_for_mdot
# -------------------------------------------------------------------
def solve_ode_for_mdot(m_dot_test: float) -> float:
    """
    Solve the ODE for a given total mass flow rate and return the final temperature.

    Parameters
    ----------
    m_dot_test : float
        Total mass flow rate (kg/s)

    Returns
    -------
    T_final : float
        Final bulk temperature (K)
    """
    params_i = (
        m_b,           # Mass of battery (kg)
        C_b,           # Battery heat capacity (J/kg·K)
        I_0,           # Current (A)
        DC_IR * 240,   # Resistance for 240 cells (Ω)
        A_s,           # Wetted area (m²)
        T_in,          # Coolant inlet temperature (K)
        m_dot_test     # Mass flow rate (kg/s)
    )

    t0 = 0
    T0 = T_in
    t_final = q_b / I_0

    sol = solve_ivp(
        fun=lambda t, T, *_: dTb_dt_sensitivity(T, t, params_i),
        t_span=[t0, t_final],
        y0=[T0],
        method='LSODA',
        rtol=1e-6,
        atol=1e-8
    )

    return sol.y[0][-1]


# -------------------------------------------------------------------
# Function: run_sensitivity_analysis
# -------------------------------------------------------------------
def run_sensitivity_analysis():
    """
    Perform mass flow rate sensitivity analysis and plot
    maximum battery temperature as a function of flow rate.
    """
    rho_avg = 680  # density of n-heptane at room temp (kg/m³)
    m_dot_test_kg_s = np.linspace(0.005, 0.1, 30)
    Q_L_min = m_dot_test_kg_s / rho_avg * 60000  # L/min

    T_max_K = []

    print(f"--- Running Sensitivity Analysis (I = {I_0} A) ---")

    for m_dot in m_dot_test_kg_s:
        try:
            T_final = solve_ode_for_mdot(m_dot)
            T_max_K.append(T_final)
            print(f"Flow: {m_dot:.5f} kg/s -> T_max: {T_final:.2f} K")
        except Exception as e:
            T_max_K.append(np.nan)
            print(f"Flow: {m_dot:.5f} kg/s -> ERROR: {e}")

    T_max_C = np.array(T_max_K) - 273.15
    T_max_constraint = T_b_max - 273.15

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(
        Q_L_min, T_max_C,
        label=f'Max Battery Temp. at I={I_0} A',
        color='red',
        linewidth=3
    )
    plt.axhline(
        y=T_max_constraint,
        color='blue',
        linestyle='--',
        label=f'Thermal Constraint ({T_max_constraint:.0f} °C)'
    )

    plt.xlabel('Total Mass Flow Rate, $\\dot{m}$ (L/min)', fontsize=14)
    plt.ylabel('Maximum Battery Temperature, $T_{b, max}$ (°C)', fontsize=14)
    plt.title(
        'Sensitivity of Maximum Battery Temperature to Coolant Flow Rate', fontsize=16
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.ylim(T_max_C.min() - 2, T_max_C.max() + 2)
    plt.tight_layout()
    plt.show()

    print("\n--- Analysis Complete ---")
    print("Plot saved as T_max_vs_MassFlowRate_Sensitivity.png")


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_sensitivity_analysis()
