# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:30:54 2023

@author: Sebastiano Tomasi
"""

import numpy as np
from math import sqrt, exp

import simulation_parameters as par

import sys
sys.path.append('C:/Users/sebas/custom_libraries')
import numerical_methods as nm  # Assuming this contains your custom integrate function

def matter_density_evolution_a(a):
    """Matter density parameter as a function of scale factor a."""
    return par.omega_m0 / a**3

def radiation_density_evolution_a(a):
    """Radiation density parameter as a function of scale factor a."""
    return par.omega_r0 / a**4

def curvature_density_evolution_a(a):
    """Curvature density parameter as a function of scale factor a."""
    return par.omega_k0 / a**2  # Assuming omega_k0 includes necessary units

def solve_dark_density_evolution_numerical_a():
    """
    Computes the dark energy density evolution numerically.

    Returns:
        numpy.ndarray: An array containing scale factor values and corresponding dark energy density.
    """
    # Define x = ln(a)
    x_min = np.log(par.bkg_a_min)
    x_max = np.log(par.bkg_a_max)
    
    # Integrate to compute the exponent in dark energy density evolution
    def de_exponent(x):
        """Exponent in the dark energy density evolution integral."""
        a = np.exp(x)
        return -3 * (1 + par.de_eos_a(a))
    
    # Perform the integration using custom integrate function
    de_exponent_integral = nm.integrate(
        f=de_exponent,
        a=x_min,
        b=x_max,
        Fa=0.0,
        rtol=par.dark_energy_density_evolution_rtol,
        atol=par.dark_energy_density_evolution_atol,
        max_step=par.dark_energy_density_evolution_max_step
    )
    
    x_values = de_exponent_integral[0]
    integral_values = de_exponent_integral[1]
    
    a_values = np.exp(x_values)
    omega_de_values = par.omega_l0 * np.exp(integral_values)
    
    return np.array([a_values, omega_de_values])

def create_log_friedmann_equation_integrand(dark_energy_density_evolution_a):
    """
    Creates the integrand function for the Friedmann equation in terms of y = ln(a).
    
    Args:
        dark_energy_density_evolution_a (callable): Function returning dark energy density as a function of 'a'.
    
    Returns:
        callable: Integrand function for the Friedmann equation.
    """
    E = create_rescaled_hubble_function_a(dark_energy_density_evolution_a)
    def friedmann_equation_integrand(y):
        """Integrand of Friedmann equation in terms of y = ln(a)."""
        a = np.exp(y)
        return 1 / E(a)
    return friedmann_equation_integrand

def create_rescaled_hubble_function_a(dark_energy_density_evolution_a):
    """
    Creates the rescaled Hubble function E(a) = H(a) / H0.
    
    Args:
        dark_energy_density_evolution_a (callable): Function returning dark energy density as a function of 'a'.
    
    Returns:
        callable: Rescaled Hubble function E(a).
    """
    def rescaled_hubble_function_a(a):
        omega_m = matter_density_evolution_a(a)
        omega_r = radiation_density_evolution_a(a)
        omega_de = dark_energy_density_evolution_a(a)
        omega_k = curvature_density_evolution_a(a)
        return np.sqrt(omega_m + omega_r + omega_de + omega_k)
    return rescaled_hubble_function_a

def z_a(a):
    """Redshift as a function of scale factor a."""
    return 1 / a - 1

def a_z(z):
    """Scale factor as a function of redshift z."""
    return 1 / (1 + z)
