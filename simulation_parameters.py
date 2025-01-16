# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:11:06 2023

@author: Sebastiano Tomasi
"""

import numpy as np
import sys, os,warnings
sys.path.append('C:/Users/sebas/Documents/GitHub/custom_libraries')
import numerical_methods as nm
import constants as const

#%% Output and Plot Settings
"""If true, it loads the saved data for the entanglement entropy and displays them.
To select which data to load, set the mass (mu) and the cosmology. The other parameters will 
be imported automatically."""
plot_saved_data = True  # True or False

"""Output saving options."""
save_plots = True  # True or False

save_data = save_plots # True or False
save_plot_dir = "./plots"

#%% Physical Constants

#For planck units
# G = 1
# c = 1  
# hbar = 1  

#For natural units
G = 6.718e-45#MeV
c = 1  
hbar = 1 
# l_planck=1e-19 

#For SI units
# G = const.G
# c = const.c
# hbar = const.hbar




# %%
"""Those are the most commons parameters that I usually change"""
N_t =200  # Number of time points to consider
N = 30 # Number of considered spherical shells
num_n_val=30
l_max = 500 # l_max is the maximum l in the spherical harmonic expansion of the field.
mu = 0 # Field mass in MeV, electron has 0.5MeV

#%% Cosmology Setup

cosmologies = ["flat", "eds", "lcdm", "rad", "ds", "snyder"]
cosmology = "snyder"

"""We leave the possibility of defining a custom equation of state for dark energy. 
Here you can write your own. Default is the cosmological constant."""
def de_eos_a(a):
    """Dark Energy Equation of State as a function of the scale parameter a."""
    return -1

H0 = 67  # Km/(s*Mpc)
ds_hubble_constant=0.5
hubble_constant = H0 * const.convert_hubble  # It is in Gy^-1. Carefull if you use non SI units. Fon snyder does not matter since hubble is not used.

"""Used to compute the time at wich the adiabatic condition does not hold anymore: H(t)=max_h_for_adiabaticity"""
max_h_for_adiabaticity=-1e-2



#%%
"""Parameters of the collapsing star, used in the Snyder collapse model."""

# rho_nuclear = 2.3e17    # Nuclear density in kg/m^3
# # Mass of the collapsing star
# bh_mass = 10 * const.Msun    # Mass in kg
# # Initial radius of the collapsing star (r_b) such that initial density equals nuclear density
# r_b = ((3 * bh_mass) / (4 * np.pi * rho_nuclear))**(1/3)  # Radius in meters
# # Schwarzschild radius (r_s)
# r_s = (2 * G * bh_mass) / c**2  # Radius in meters

r_s = 1#In natural units this is 1/MeV=1.97327e−13 meters
r_b = (r_s*8)**(1/3)

k = r_s / r_b**3  # Spatial curvature
collapse_time = np.pi / (2 * c * np.sqrt(k))  # Comoving time for the collapse

# t_rs is the time it takes for the collapsing sphere to reach the Schwarzschild radius
t_rs = (1 / (2 * c * np.sqrt(k))) * (
    2 * np.sqrt(r_s * (r_b - r_s) / r_b**2) +
    np.arccos((2 * r_s / r_b) - 1)
)


#%% Time Settings
"""Times at which we compute the entanglement entropy"""

if cosmology=="snyder":
    t_ini = 0  # Time at which the initial conditions on the ground state are imposed.
    t_min = t_ini  # First time at which the entropy scaling is computed
    t_max = t_min + collapse_time * (1 - 1e-2)  # Last time at which the entropy scaling is computed
    # t_max = t_min + t_rs/10
elif cosmology=="ds":
    t_ini = -1  # Time at which the initial conditions on the ground state are imposed.
    t_min = t_ini  # First time at which the entropy scaling is computed
    t_max=1
else:
    t_ini = 0.1  # Time at which the initial conditions on the ground state are imposed.
    t_min = t_ini  # First time at which the entropy scaling is computed
    t_max=1
    
logspaced_times = False  # Use log-spaced time points

#%% Spatial Settings
"""The oscillators are traced out in the spherical shell between n_min and n_max = n_min + N."""
"""N is the size of the covariance matrix Σ^l. We can either fix N, to which corresponds
a horizon size H0_c, or fix H0_c which in turn fixes N."""
n_min = 0  # First considered shell is at n_min.


if cosmology == "snyder":
    cut_off=r_b/(n_min+N)
    # cut_off=1e-9
    H0_c = cut_off * (n_min+N)  # Comoving size of the horizon, fixed by the initial radius of the star
else:
    # cut_off = 8.2e-23 # Value of the comoving cut off = l_planc in 1/MeV.
    cut_off=H0_c/(n_min+N)
    H0_c = cut_off * (n_min+N)

"""Since the computations are very demanding, we choose the sizes of the inside system that
we want to trace out. This is useful, for example, if the system is known to satisfy an area law,
then we can just compute the entanglement entropy for one subsystem of size N // n, and from this
we can compute the slope of the area law.
For example, if we set n_values = np.arange(n_min, n_max + 1) we have maximum information
on the entropy scaling. This can be used to check if the system satisfies an area law.
But we can diminish the number of n_values for sake of speed. For example, we may consider
only half of the points, n_values = np.arange(n_min, n_max + 1, 2), or just two points n_values = [N // 4, N // 2]."""
n_values = n_min + np.asarray(sorted(set([
    int(i / num_n_val * N)
    for i in range(num_n_val+1)
    if 1 - i / num_n_val > 0.15
])))

# n_values = n_min+np.asarray([n_min, int(23/20 * N)])

# n_values = n_min+np.arange(0, 20*N//23+1, 1)


# %% CONSTRUCT DIRECTORY PATH
def round_to_significant_digits(value, significant_digits):
    if value == 0:
        return 0
    else:
        return round(value, significant_digits - int(f"{value:.1e}".split('e')[1]) - 1)
    
fixed_name_left=f"{save_plot_dir}/{cosmology}/mu={mu}/cut_off={round_to_significant_digits(cut_off,4)}/"
# fixed_name_left=f"{save_plot_dir}/{cosmology}/mu={mu}/area_law_holds/"

fixed_name_right=f""
os.makedirs(fixed_name_left, exist_ok=True)


#%% Precision Parameters
"""Here you can set the precision parameters for the various integrations that the code performs."""

"""Particle horizon integration parameters"""
# hor_atol = 1e-15
# hor_rtol = 1e-12
hor_atol = 1e-9
hor_rtol = 1e-6

horizon_a_min = 1e-9
horizon_a_max = 1
horizon_size_at_a_min = 0

"""Background integration limits, used for the dark density evolution and Friedmann equation integration"""
bkg_a_min = 1e-12
if cosmology=="snyder":
    bkg_a_max = 1-bkg_a_min
else:
    bkg_a_max = 1

"""Precision parameters for the integral of the fluid equation."""
dark_energy_density_evolution_atol = 1e-11
dark_energy_density_evolution_rtol = 1e-8
dark_energy_density_evolution_max_step =np.inf# 1e-3

"""Precision parameters for the integral of the Friedmann equation to compute t(a)."""
friedmann_atol = 1e-12
friedmann_rtol = 1e-9
# friedmann_max_stepsize = 1e-3
friedmann_max_stepsize=np.inf

"""Ermakov-like equation integration
These have a great impact on the code performance. While the cosmology is solved
once for all, we have to solve the Ermakov-like equation np.sum(l_max * n_values) * len(times) times.
If the code runs slow, you can diminish the precision."""
# ermak_atol = 1e-12
# ermak_rtol = 1e-10
ermak_atol = 1e-10
ermak_rtol = 1e-8
# ermak_atol = 1e-8
# ermak_rtol = 1e-6
# ermak_atol = 1e-6
# ermak_rtol = 1e-4

"""We have to set it to true or otherwise we get negetive values for the 
frequency squared."""
use_midpoint_scheme = True


#%% Debug and Warnings
"""Displays warnings and debug information"""

"""The higher verbose is, the more warnings the code prints."""
verbose = 2
debug_level = 4

"""For each value of the debug level, we print everything it 
prints with lower debug levels.

if debug_level == 0:
    - Print nothing.
if debug_level == 1:
    - Print l in the l-cycle. 
if debug_level == 2:
    - Plot the background cosmology quantities. 
if debug_level == 3:
    - Plot some solutions for rho_lj(t). 
if debug_level == 4:
    - Plot even more solutions for rho_lj(t).
    - Print the % error on the entanglement entropy."""


#%% Density Parameters for the Chosen Cosmology
cosmology_params = {
    "eds":    {"omega_m0": 1.0, "omega_r0": 0.0, "omega_k0": 0.0},
    "lcdm":   {"omega_m0": 0.3, "omega_r0": 0.0, "omega_k0": 0.0},
    "rad":    {"omega_m0": 0.0, "omega_r0": 1.0, "omega_k0": 0.0},
    "ds":     {"omega_m0": 0.0, "omega_r0": 0.0, "omega_k0": 0.0},
    "flat":   {"omega_m0": 1.0, "omega_r0": 0.0, "omega_k0": 0.0},
    "snyder": {"omega_m0": 1.0, "omega_r0": 0.0, "omega_k0": 0.0, "omega_l0": 0.0},
}

# Retrieve the parameters for the given cosmology
params = cosmology_params.get(cosmology)

if params:  # It means: if params is not None and not empty
    omega_m0 = params.get("omega_m0", 0.0)
    omega_r0 = params.get("omega_r0", 0.0)
    omega_k0 = params.get("omega_k0", 0.0)

    # Calculate omega_l0 if it's not explicitly provided
    omega_l0 = params.get("omega_l0", 1.0 - omega_m0 - omega_r0 + omega_k0)
else:
    raise ValueError(f"Unknown cosmology type: {cosmology}")

#%% Derived Parameters
"""cut_off_type and horizon_type  cannot be changed because the other options are not yet fully implemented."""

cut_off_types = ["fixed_comoving", "fixed_physical"]
cut_off_type = "fixed_comoving"
horizon_types = ["fixed_comoving", "fixed_physical", "particle_horizon"]
horizon_type = "fixed_comoving"

if logspaced_times:
    times = np.round(nm.logspace(t_min, t_max, N_t), 4)
else:
    times = np.round(np.linspace(t_min, t_max, N_t), 4)

n_max = n_min + N

#%% CHECKS AND WARNINGS

if not use_midpoint_scheme:
    warnings.warn("Midpoint scheme is not being used. This will lead to negative \Gamma^2 values, which are unphysical!")


if n_max * cut_off > r_b:
    raise Exception(f"Cannot trace out degrees of freedom outside the collapsing sphere, "
                    f"n_max * cut_off = {n_max * cut_off} must be smaller than r_b = {r_b}")

if r_b<r_s:
    raise Exception(f"The boundary radius of the dust sphere r_b={r_b} must be smaller then the Schwarzschild radius r_s={r_s}")
if r_b/cut_off<n_max:
    raise Exception(f"r_b/cut_off={r_b/cut_off} must be greater then n_max={n_max} in order to have a well defined metric (no sign changes)")
    
if max(n_values)>N:
    raise Exception(f"Cannot trace out {max(n_values)} oscillators since there are a total of {N}.\n Modify n_values={n_values} accrodingly.")

# Check if t_ini is smaller than t_min. Raise error otherwise
if t_ini > t_min:
    raise Exception("t_min must be greater than t_ini.")
# Insert t_ini in the times list, as the first entry
times = np.insert(times, 0, t_ini)

save_data = False  if plot_saved_data else save_plots  # True or False

# %% IMPORTANT PARAMEERS


"""The important parameters are printed when the compute_entanglement_entropy function is called. 
The values of those parameters are also saved in a .txt file in the plot folder."""

important_parameters = ["cosmology","n_min", "N","num_n_val","n_values", "l_max", 
                        "cut_off", 
                        "mu",
                        "t_min", "t_max", "N_t",
                        ]

if cosmology=="snyder":
    important_parameters.extend(["k", "r_b", "r_s", "t_rs", "collapse_time"])
elif cosmology=="ds":
    important_parameters.extend(["ds_hubble_constant"])
else:
    important_parameters.extend(["omega_m0", "omega_r0", "omega_l0", "omega_k0"])
    

#%% Print Variables at the End
if __name__ == "__main__":
    print("\nOutput and Plot Settings:")
    print("  Load saved data (plot_saved_data) =", plot_saved_data)
    print("  Save data (save_data) =", save_data)
    print("  Save plots (save_plots) =", save_plots)
    print("  Plots directory (fixed_name_left) =", fixed_name_left)

    print("\nPhysical Constants:")
    print("  Gravitational constant (G) =", G)
    print("  Speed of light (c) =", c)
    print("  Reduced Planck constant (hbar) =", hbar)

    print("\nCosmology Settings:")
    print("  Selected cosmology =", cosmology)
    print("  Hubble constant (H0) =", H0)
    print("  Hubble constant in Gy^-1 (hubble_constant) =", hubble_constant)

    print("\nCollapsing Star Parameters:")
    print("  Schwarzschild radius (r_s) =", r_s)
    print("  Initial radius of star (r_b) =", r_b)
    print("  Spatial curvature (k) =", k)
    print("  Collapse time (collapse_time) =", collapse_time)
    print("  Time to reach Schwarzschild radius (t_rs) =", t_rs)

    print("\nTime Settings:")
    print("  Initial time (t_ini) =", t_ini)
    print("  Start time (t_min) =", t_min)
    print("  End time (t_max) =", t_max)
    print("  Number of time points (N_t) =", N_t)
    print("  Use log-spaced times (logspaced_times) =", logspaced_times)

    print("\nSpatial Settings:")
    print("  Minimum shell index (n_min) =", n_min)
    print("  Number of shells (N) =", N)
    print("  Cut-off (cut_off) =", cut_off)
    print("  Horizon size (H0_c) =", H0_c)
    print("  Traced shell indices (n_values) =", n_values)
    print("  Maximum l in spherical harmonic expansion (l_max) =", l_max)
    print("  Field mass (mu) =", mu)

    print("\nPrecision Parameters:")
    print("  Horizon integration absolute tolerance (hor_atol) =", hor_atol)
    print("  Horizon integration relative tolerance (hor_rtol) =", hor_rtol)
    print("  Horizon a_min (horizon_a_min) =", horizon_a_min)
    print("  Horizon a_max (horizon_a_max) =", horizon_a_max)
    print("  Horizon size at a_min (horizon_size_at_a_min) =", horizon_size_at_a_min)
    print("  Background a_min (bkg_a_min) =", bkg_a_min)
    print("  Background a_max (bkg_a_max) =", bkg_a_max)
    print("  Dark energy density evolution atol =", dark_energy_density_evolution_atol)
    print("  Dark energy density evolution rtol =", dark_energy_density_evolution_rtol)
    print("  Dark energy density evolution max step =", dark_energy_density_evolution_max_step)
    print("  Friedmann equation atol (friedmann_atol) =", friedmann_atol)
    print("  Friedmann equation rtol (friedmann_rtol) =", friedmann_rtol)
    print("  Friedmann equation max step size =", friedmann_max_stepsize)
    print("  Ermakov equation atol (ermak_atol) =", ermak_atol)
    print("  Ermakov equation rtol (ermak_rtol) =", ermak_rtol)
    print("  Use midpoint discretization scheme (use_midpoint_scheme) = ",use_midpoint_scheme)

    print("\nDebug and Warnings:")
    print("  Verbose level (verbose) =", verbose)
    print("  Debug level (debug_level) =", debug_level)

    print("\nDerived Parameters:")
    print("  Cut-off type (cut_off_type) =", cut_off_type)
    print("  Horizon type (horizon_type) =", horizon_type)
    print("  Maximum shell index (n_max) =", n_max)

