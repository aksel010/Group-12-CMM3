# model_runner.py
import ODE
import Mass_flowrate
import Optimum_Current
import RK4_Error

def run_simulation(config):
    # get all central results as main.py
    ode_data = ODE.run()
    mf_data = Mass_flowrate.run()
    oc_data = Optimum_Current.run()
    RK4_Error.run()  # Populate log if needed

    # Compose log string with convergence details and results
    log = "Simulation complete.\n"
    log += f"Optimum Current: {config['I_0']}\n"
    log += f"Mass Flow Rate Sample: {mf_data['mass_flow'][0]}\n"

    return {
        "ode": ode_data,
        "mass_flow": mf_data['mass_flow'],
        "residuals": mf_data['residuals'],
        "optimum": {
            "I_smooth": oc_data['smooth'][0],
            "delta_T_smooth": oc_data['smooth'][1],
            "critical_current": oc_data['critical'][0]
        },
        "log": log
    }
