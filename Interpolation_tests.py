import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def import_nist_txt(filename='n heptane.txt'):
    """
    Import NIST data from text file
    The file has 14 columns, we only need 6 of them.
    """
    df = pd.read_csv(filename, sep=r'\s+', skiprows=1, header=None)
    
    column_map = {
        0: 'T_K',
        1: 'Pressure',
        2: 'Density',       
        8: 'Cp',          
        12: 'Thermal_Cond',
        11: 'Viscosity' 
    }
    
    df = df.rename(columns=column_map)[list(column_map.values())]
    
    M = 0.100205 # kg/mol
    
    #unit conversions
    df['Density'] = df['Density'] * 1000 * M 
    df['Cp'] = df['Cp'] / M
    df['Viscosity'] = df['Viscosity'] * 1e-6   
    
    return df

nist_df = import_nist_txt('n heptane.txt') 
print(nist_df.head())