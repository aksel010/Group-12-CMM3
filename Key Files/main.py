"""
main.py - PHEV BTMS Optimization
=================================
Design Question: Determine minimal charging time by varying Current (I) 
and mass flow-rate (ṁ) while keeping T_max ≤ 40°C
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Import from existing files (DO NOT MODIFY THEM)
from config import q_b, m_b, C_b, T_in, T_b_max, R_b
from ODE import Tb, dTb_dt

# ============================================================================
# CONFIGURATION
# ============================================================================

# Design space
CURRENT_RANGE = np.arange(5, 80, 5)           # A
FLOW_RATE_RANGE = np.arange(0.001, 0.05, 0.005)  # kg/s
A_S = 0.01  # Heat transfer area [m²]

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def simulate_point(I, m_dot):
    """Simulate one (I, ṁ) design point. Returns dict with results."""
    
    # Setup parameters for ODE solver
    params = (m_b, C_b, I, R_b, A_S, T_in, m_dot)
    
    try:
        # Run thermal simulation
        t, T = Tb(dTb_dt, params, stepsize=1.0)
        
        # Results
        t_charge = q_b / I
        T_max = np.max(T)
        feasible = T_max <= T_b_max
        
        return {
            'I': I,
            'm_dot': m_dot,
            't_charge': t_charge,
            'T_max': T_max,
            'T_max_C': T_max - 273.15,
            'feasible': feasible,
            't': t,
            'T': T
        }
    except:
        return {'I': I, 'm_dot': m_dot, 'feasible': False}


def run_optimization():
    """Run 2D grid search over (I, ṁ)"""
    
    print("="*70)
    print(" PHEV BTMS OPTIMIZATION")
    print("="*70)
    print(f" Current range: {CURRENT_RANGE[0]}-{CURRENT_RANGE[-1]} A ({len(CURRENT_RANGE)} points)")
    print(f" Flow rate range: {FLOW_RATE_RANGE[0]:.3f}-{FLOW_RATE_RANGE[-1]:.3f} kg/s ({len(FLOW_RATE_RANGE)} points)")
    print(f" Total simulations: {len(CURRENT_RANGE) * len(FLOW_RATE_RANGE)}")
    print(f" Constraint: T_max ≤ {T_b_max - 273.15:.0f}°C")
    print("-"*70)
    
    # Run all simulations
    results = []
    start = time.time()
    
    for i, I in enumerate(CURRENT_RANGE):
        print(f"Progress: {i+1}/{len(CURRENT_RANGE)} currents... ", end='\r')
        for m_dot in FLOW_RATE_RANGE:
            results.append(simulate_point(I, m_dot))
    
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")
    
    # Find optimal
    feasible = [r for r in results if r['feasible']]
    
    if not feasible:
        print("\n⚠️  NO FEASIBLE SOLUTIONS - Increase flow rate or reduce current")
        return results, None
    
    optimal = min(feasible, key=lambda x: x['t_charge'])
    
    # Print results
    print("\n" + "="*70)
    print(" OPTIMAL SOLUTION")
    print("="*70)
    print(f" ✓ Optimal Current:              {optimal['I']:.1f} A")
    print(f" ✓ Optimal Flow Rate:            {optimal['m_dot']:.4f} kg/s")
    print(f" ✓ Minimum Charging Time:        {optimal['t_charge']:.0f} s ({optimal['t_charge']/60:.1f} min)")
    print(f" ✓ Maximum Temperature:          {optimal['T_max_C']:.2f}°C")
    print(f" ✓ Temperature Margin:           {T_b_max - optimal['T_max']:.2f} K")
    print(f" ✓ Feasible Solutions:           {len(feasible)}/{len(results)}")
    print("="*70)
    
    return results, optimal


def plot_results(results, optimal):
    """Create visualization"""
    
    # Extract data
    feasible = [r for r in results if r.get('feasible', False)]
    if not feasible:
        return
    
    I = np.array([r['I'] for r in feasible])
    m = np.array([r['m_dot'] for r in feasible])
    t = np.array([r['t_charge']/60 for r in feasible])
    T = np.array([r['T_max_C'] for r in feasible])
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Temperature map
    sc1 = axes[0].scatter(I, m, c=T, cmap='RdYlBu_r', s=100, edgecolors='k', linewidth=0.5)
    axes[0].scatter(optimal['I'], optimal['m_dot'], marker='★', s=500, 
                    c='gold', edgecolors='black', linewidth=2, zorder=10)
    axes[0].axhline(T_b_max - 273.15, color='red', ls='--', lw=2)
    axes[0].set_xlabel('Current [A]', fontweight='bold')
    axes[0].set_ylabel('Flow Rate [kg/s]', fontweight='bold')
    axes[0].set_title('Max Temperature [°C]', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[0], label='T_max [°C]')
    
    # Plot 2: Charging time map
    sc2 = axes[1].scatter(I, m, c=t, cmap='viridis', s=100, edgecolors='k', linewidth=0.5)
    axes[1].scatter(optimal['I'], optimal['m_dot'], marker='★', s=500, 
                    c='gold', edgecolors='black', linewidth=2, zorder=10)
    axes[1].set_xlabel('Current [A]', fontweight='bold')
    axes[1].set_ylabel('Flow Rate [kg/s]', fontweight='bold')
    axes[1].set_title('Charging Time [min]', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=axes[1], label='Time [min]')
    
    # Plot 3: Optimal temperature profile
    axes[2].plot(optimal['t']/60, optimal['T'] - 273.15, 'b-', linewidth=2)
    axes[2].axhline(T_b_max - 273.15, color='red', ls='--', lw=2, label='T_max limit')
    axes[2].fill_between(optimal['t']/60, T_b_max - 273.15, 50, alpha=0.2, color='red')
    axes[2].set_xlabel('Time [min]', fontweight='bold')
    axes[2].set_ylabel('Temperature [°C]', fontweight='bold')
    axes[2].set_title(f'Optimal Solution (I={optimal["I"]:.1f}A)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('BTMS_optimization_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plots saved to 'BTMS_optimization_results.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Run optimization
    results, optimal = run_optimization()
    
    # Plot results
    if optimal:
        plot_results(results, optimal)
    
    print("\n✓ Optimization complete!\n")
