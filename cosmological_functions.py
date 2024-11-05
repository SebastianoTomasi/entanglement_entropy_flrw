# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:30:54 2023

@author: Sebastiano Tomasi
"""


import numpy as np
from numpy import sqrt,exp


import simulation_parameters as par

import sys
sys.path.append('C:/Users/sebas/custom_libraries')
import numerical_methods as nm


#%%

def matter_density_evolution_a(a):
    """Matter density parameter in function of a"""
    matter_density_evolution_a= par.omega_m0/a**3
    return matter_density_evolution_a


def radiation_density_evolution_a(a):
    """Radiation density parameter in function of a"""
    radiation_density_evolution_a= par.omega_r0/a**4
    return radiation_density_evolution_a

def curvature_density_evolution_a(a):
    """Curvature density parameter in function of a"""
    curvature_density_evolution_a=par.omega_k0*par.c**2/a**2
    return curvature_density_evolution_a


def solve_dark_density_evolution_numerical_a():
    """Finds the function g(a) for the dark energy.
    Definition of the integrand in the exponent."""
    def de_exponent(x):
        """Since the integral is done w.r.t dx, x=ln(a), the expo"""
        de_exponent = -3*(1+par.de_eos_a(np.exp(x)))
        return de_exponent
    """Compute dark_energy_density_evolution_a(a)=omega_l0*g(a) """
    de_exponent_integral=nm.integrate(f=de_exponent,a=np.log(par.bkg_a_min),b=np.log(par.bkg_a_max),
                                        Fa=0,rtol=par.dark_energy_density_evolution_atol,
                                        atol=par.dark_energy_density_evolution_rtol,
                                        max_step=par.dark_energy_density_evolution_max_step)
    scale_parameter_values=np.exp(de_exponent_integral[0])
    dark_density_evolution_numerical_a=np.array([scale_parameter_values,par.omega_l0*exp(de_exponent_integral[1])])
    
    return dark_density_evolution_numerical_a

"""Create the integrand for computing the cosmological time, also in d[log(a)]"""
def create_friedmann_equation_integrand(dark_energy_density_evolution_a):#Not used 
    E=create_rescaled_hubble_function_a(dark_energy_density_evolution_a)
    def friedmann_equation_integrand(a):
        return 1/(a*E(a))
    return friedmann_equation_integrand

def create_log_friedmann_equation_integrand(dark_energy_density_evolution_a):#Logarithmic integrand
    E=create_rescaled_hubble_function_a(dark_energy_density_evolution_a)
    def friedmann_equation_integrand(y):#take as input y=ln(a)
        return 1/E(np.exp(y))
    return friedmann_equation_integrand

def create_rescaled_hubble_function_a(dark_energy_density_evolution_a):
    """The rescaled hubble function is usually denoted by E(a)=(H/H0)."""
    def rescaled_hubble_function_a(a):
        return sqrt(matter_density_evolution_a(a) + dark_energy_density_evolution_a(a)+radiation_density_evolution_a(a)+curvature_density_evolution_a(a))
    return rescaled_hubble_function_a



"""Conversion between redshift <-> scale parameter"""
def z_a(a):
    z_a=1/a-1
    return z_a
def a_z(z):
    a_z=1/(1+z)
    return a_z