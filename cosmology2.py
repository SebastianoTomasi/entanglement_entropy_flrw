# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:30:54 2023

@author: Sebastiano Tomasi
"""

import numpy as np
from scipy import interpolate
from math import sqrt
import warnings


import simulation_parameters as par
import cosmological_functions2 as cosm_func

import sys
sys.path.append('C:/Users/sebas/custom_libraries')
import plotting_function as pl
# Ensure that 'integrate' function is available
from numerical_methods import integrate as custom_integrate


# Private flag to track initialization
_initialized = False

def compute_cosmology_functions():
    """Compute cosmological functions based on the selected cosmology."""

    # Supported cosmologies
    supported_cosmologies = ["eds", "lcdm", "rad", "ds", "snyder", "flat"]

    if par.cosmology not in supported_cosmologies:
        raise ValueError(f"Cosmology '{par.cosmology}' not recognized. Supported cosmologies are {supported_cosmologies}.")

    # Initialize functions
    scale_factor_t = None
    hubble_function_t = None
    comoving_horizon_t = None
    cut_off_t = None
    universe_age = None

    # General FLRW cosmology case
    if par.cosmology in ["eds", "lcdm", "rad"]:
        # Solve dark energy density evolution numerically
        dark_density_evolution_numerical_a = cosm_func.solve_dark_density_evolution_numerical_a()
        a_values = dark_density_evolution_numerical_a[0]
        dark_density_values = dark_density_evolution_numerical_a[1]

        # Interpolate dark density evolution
        dark_density_evolution_a = interpolate.InterpolatedUnivariateSpline(a_values, dark_density_values, k=3)

        # Define the integrand for the Friedmann equation
        def friedmann_integrand_log(a):
            E = cosm_func.create_rescaled_hubble_function_a(dark_density_evolution_a)
            return 1 / E(np.exp(a))

        # Integrate to get H0 * t(a)
        log_a_min = np.log(par.bkg_a_min)
        log_a_max = np.log(par.bkg_a_max)

        hubble_constant_times_t = custom_integrate(
            f=friedmann_integrand_log,
            a=log_a_min,
            b=log_a_max,
            Fa=0.0,
            rtol=par.friedmann_rtol,
            atol=par.friedmann_atol,
            max_step=par.friedmann_max_stepsize
        )

        # Extract scale factor and time values
        log_a_values = hubble_constant_times_t[0]
        a_values = np.exp(log_a_values)
        t_values = hubble_constant_times_t[1] / par.hubble_constant
        
        # Create interpolation functions
        scale_factor_t = interpolate.InterpolatedUnivariateSpline(t_values, a_values, k=3)
        cosmic_time_a = interpolate.InterpolatedUnivariateSpline(a_values, t_values, k=3)

        # Compute derivative da/dt
        da_dt = scale_factor_t.derivative()(t_values)

        # Compute Hubble function H(t) = (da/dt)/a
        H_values = da_dt / a_values
        hubble_function_t = interpolate.InterpolatedUnivariateSpline(t_values, H_values, k=3)
        
        

        # Compute the age of the universe at a = 1
        universe_age = cosmic_time_a(1.0)
        if __name__ == "__main__":
            print(f"Age of the universe = {universe_age}")

        # Plotting scale factor a(t)
        if __name__ == "__main__" or par.debug_level >= 2:
            # Analytical solution for EdS universe
            t_analytical = t_values
            a_analytical = (1.5 * par.hubble_constant * t_analytical) ** (2 / 3)

            pl.plot([
                [t_values, a_values],
                [t_analytical, a_analytical]],
                title="Scale Factor a(t)",
                xlabel="Time t",
                ylabel="a(t)",
                legend=["Numerical", "EdS Analytical"],
            )
            
            # Analytical solution for EdS universe
            t_analytical = t_values
            H_analytical = 1 / (2*t_analytical)

            pl.plot([
                [t_values, H_values],
                [t_analytical, H_analytical]],
                title="Hubble function H(t)",
                xlabel="Time t",
                ylabel="H(t)",
                yscale="log",xscale="log",dotted=True,connected_dots=True,
                legend=["Numerical", "EdS Analytical"]
            )

        # Compute comoving horizon
        comoving_horizon_t = compute_comoving_horizon(scale_factor_t, cosmic_time_a, dark_density_evolution_a)

        # Define cutoff function
        cut_off_t = define_cutoff_function(scale_factor_t)

    elif par.cosmology == "ds":
        # Issue the warning
        warnings.warn("Not fully implemented but works in the basic cases.", UserWarning)
        # De Sitter universe
        scale_factor_t = lambda t: np.exp(par.ds_hubble_constant * t)
        hubble_function_t = lambda t: par.ds_hubble_constant
        comoving_horizon_t = lambda t: par.H0_c
        cut_off_t = lambda t: par.cut_off
        t=np.linspace(par.t_ini, par.t_max,200)
        
        if __name__ == "__main__" or par.debug_level >= 2:
            pl.plot([t,scale_factor_t(t)],
                    title="Scale Factor a(t)",
                    xlabel="Time t",
                    ylabel="a(t)"
                    )

    elif par.cosmology == "snyder":
        # Oppenheimer-Snyder collapse model
    
        # Define the logarithmic integrand
        def friedmann_equation_log_integrand(y):
            a=np.exp(y)
            return -a/np.sqrt(1/a-1)
    
        # Set the integration limits in terms of y = ln(a)
        log_a_min = np.log(par.bkg_a_min)
        log_a_max = np.log(par.bkg_a_max)
    
        # Perform the integration using your custom integrate function
        constants_times_t = custom_integrate(
            f=friedmann_equation_log_integrand,
            a=log_a_min,
            b=log_a_max,
            Fa=0.0,
            rtol=par.friedmann_rtol,
            atol=par.friedmann_atol,
            max_step=par.friedmann_max_stepsize
        )
    
        # Extract scale factor and time values
        y_values = constants_times_t[0]
        a_values = np.exp(y_values)
        t_values = constants_times_t[1] / (par.c * np.sqrt(par.k))
    
        # Shift time so that collapse occurs at t = 0
        t_values -= t_values[-1]
        
        cosmic_time_a = interpolate.InterpolatedUnivariateSpline(a_values, t_values, k=3)
        # Create interpolation functions
        
        
        # # Reverse arrays to have increasing time
        a_values = a_values[::-1]
        t_values = t_values[::-1]
        
        unique_t, unique_indices = np.unique(t_values, return_index=True)
        scale_factor_t = interpolate.InterpolatedUnivariateSpline(unique_t, a_values[unique_indices], k=3)
        
        print((len(t_values)-len(unique_indices))/len(t_values))
        
        
        
    
        # Compute derivative da/dt
        da_dt = scale_factor_t.derivative()(t_values)
    
        # Compute Hubble function H(t) = (da/dt)/a
        H_values = da_dt / a_values
        hubble_function_t = interpolate.InterpolatedUnivariateSpline(unique_t, H_values[unique_indices], k=3)
    
        # Compute collapse time
        collapse_time = t_values[-1]
        if __name__ == "__main__":
            print(f"Collapse time = {collapse_time}")
    
        # Analytical solution for comparison
        alpha = np.linspace(0, np.pi, 200)
        t_analytical = (alpha + np.sin(alpha)) / (2 * par.c * np.sqrt(par.k))
        a_analytical = (1 + np.cos(alpha)) / 2
    
        # Plotting scale factor a(t)
        if __name__ == "__main__" or par.debug_level >= 2:
            pl.plot([[t_values, a_values],
                [t_analytical, a_analytical]],
                title="Scale Factor a(t)",
                xlabel="Time t",
                ylabel="Scale Factor a(t)",
                legend=["Numerical", "Analytical"]
            )
    
        # Compute comoving horizon
        comoving_horizon_t = compute_comoving_horizon(scale_factor_t, cosmic_time_a)
    
        # Define cutoff function
        cut_off_t = define_cutoff_function(scale_factor_t)


    elif par.cosmology == "flat":
        # Flat spacetime
        scale_factor_t = lambda t: 1.0
        hubble_function_t = lambda t: 0.0
        comoving_horizon_t = lambda t: par.H0_c
        cut_off_t = lambda t: par.cut_off

    else:
        raise ValueError(f"Cosmology '{par.cosmology}' not recognized.")

    # Return computed functions
    result = {
        'scale_factor_t': scale_factor_t,
        'hubble_function_t': hubble_function_t,
        'comoving_horizon_t': comoving_horizon_t,
        'cut_off_t': cut_off_t
    }
    
    if universe_age is not None:
        result['universe_age'] = universe_age
    return result

def compute_comoving_horizon(scale_factor_t, cosmic_time_a, dark_density_evolution_a=None):
    """Compute the comoving horizon based on user choice."""
    if par.horizon_type == "fixed_comoving":
        return lambda t: par.H0_c
    elif par.horizon_type == "fixed_physical":
        return lambda t: par.H0_c / scale_factor_t(t)
    elif par.horizon_type == "particle_horizon":
        # Define the integrand for the particle horizon
        def horizon_integrand(a):
            if dark_density_evolution_a is not None:
                E = cosm_func.create_rescaled_hubble_function_a(dark_density_evolution_a)
            else:
                # For Snyder model, assuming EdS universe for simplicity
                E = lambda a_val: sqrt(par.omega_m0 / a_val**3)
            return 1 / (E(a) * a**2)

        # Integrate to get the comoving horizon as a function of 'a'
        a_min = par.horizon_a_min
        a_max = par.horizon_a_max

        horizon_integral = custom_integrate(
            f=horizon_integrand,
            a=a_min,
            b=a_max,
            Fa=par.horizon_size_at_a_min,
            rtol=par.hor_rtol,
            atol=par.hor_atol,
            max_step=par.hor_max_stepsize
        )

        a_values = horizon_integral[0]
        horizon_values = par.H0_c * horizon_integral[1]

        # Create interpolation functions
        comoving_horizon_a = interpolate.InterpolatedUnivariateSpline(a_values, horizon_values, k=3)
        comoving_horizon_t = lambda t: comoving_horizon_a(scale_factor_t(t))

        # Plotting the comoving horizon
        if __name__ == "__main__" or par.debug_level >= 2:
            a_plot = np.linspace(par.bkg_a_min, par.bkg_a_max, 200)
            pl.plot(
                [a_plot, comoving_horizon_a(a_plot)],
                [a_plot, par.H0_c * 2 * np.sqrt(a_plot)],  # Analytical solution for EdS
                title="Comoving Particle Horizon Radius",
                xlabel="Scale Factor a",
                ylabel=r"$R_H^0$",
                legend=["Numerical", "Analytical"],
                yscale="log"
            )

        return comoving_horizon_t
    else:
        raise ValueError(f"horizon_type '{par.horizon_type}' not recognized.")

def define_cutoff_function(scale_factor_t):
    """Define the cutoff function based on user choice."""
    if par.cut_off_type == "fixed_comoving":
        return lambda t: par.cut_off
    elif par.cut_off_type == "fixed_physical":
        return lambda t: par.cut_off / scale_factor_t(t)
    else:
        raise ValueError(f"cut_off_type '{par.cut_off_type}' not recognized.")



if not _initialized:
    cosmology_funcs = compute_cosmology_functions()
    # Access the computed functions
    scale_factor_t = cosmology_funcs['scale_factor_t']
    hubble_function_t = cosmology_funcs['hubble_function_t']
    comoving_horizon_t = cosmology_funcs['comoving_horizon_t']
    cut_off_t = cosmology_funcs['cut_off_t']
    try:
        universe_age = cosmology_funcs['universe_age']
    except:
        pass
    _initialized = True

