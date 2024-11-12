# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:11:06 2023

@author: Sebastiano Tomasi
"""

import numpy as np
import sys,os
sys.path.append('C:/Users/sebas/Documents/GitHub/custom_libraries')
import numerical_methods as nm
import constants as const

#%%
"""If true, it loads the saved data for the entranglement entropy and displays them.
To select which data to load, set this file as you would to run the simulation."""
plot_saved_data=False#True,False

"""Output saving options."""
save_data=True#True,False
save_data_dir="./data"
os.makedirs(save_data_dir, exist_ok=True)

save_plots=True#True,False
save_plot_dir="./plots"
os.makedirs(save_plot_dir, exist_ok=True)

#%% Physical constants
G=1#const.G
c=1#const.c
hbar=1#const.hbar

#%% Set the cosmology
cosmologies=["flat","eds","curved eds","lcdm","rad","ds","snyder"]
cosmology="snyder"

"""We leave the possibility of defining a custom equation of state for dark energy. 
Here you can write your own. Defoult is the cosmological constant."""
def de_eos_a(a):
    """DarkEnergy EquationOfState as a function of the scale parameter a."""
    return -1

H0=67#Km/(s*Mpc)
hubble_constant = H0*const.convert_hubble# It is in Gy-1

#%%
"""Parameters of the collapsing star, used in the snyder collapse model.
Whith those choiches the collapse takes 1 unit of time and the schwarzshild radius is 1/2
units of length."""

bh_mass=1/4#Mass of the collapsing star
bh_R0=(8 * bh_mass / np.pi**2)**(1/3)#Initial radius of the collapsing star.

r_s=2*G*bh_mass/c**2# Scwarzshild radius of the star
k=r_s/bh_R0**3# Spatial curvature
collapse_time=np.pi/(2*c*np.sqrt(k))# Comoving time for the collapse

#t_rs is the time it takes to the collapsing sphere to reach the schwarzshild radius
t_rs = (1 / (2 * c * np.sqrt(k))) * (
    2 * np.sqrt(r_s * (bh_R0 - r_s) / bh_R0**2) +
    np.arccos((2 * r_s / bh_R0) - 1)
)

#%%
"""Times at which we compute the entanglement entropy"""
t_ini=0# Time at wich the initial conditions on the ground state are imposed.
t_min=t_ini# First time at which the entropy scaling is computed
t_max=t_min+collapse_time*(1-1e-2)# Last time at which the entropy scaling is computed
N_t=100#Number of time points to consider
logspaced_times=False#Logspace the time points

#%%
"""The oscillators are traced out in the spherical shell between n_min and n_max=n_min+N."""
"""N is the size of the covariance matrix \Sigma^l. We can either fix N, to which correspond
an horizon size H0_c, or fix H0_c which in turn fixes N."""
n_min=0#First considered shell is at n_min. Usefull if you want to exclude a region around the origin.

# H0_c=20#Comoving size of the horizon,it fixes the number of considered spherical shells
# N=int(H0_c/cut_off)#Number of considered spherical shells

N=15#Number of considered spherical shells
cut_off=1#Value of the cut off, be aware that it is the same in comoving or physical coordinates. 
H0_c=cut_off*N#Comoving size of the horizon,it is fixed by the number of considered spherical shells

"""Sice the computations are very demanding, we choose the sizes of the inside system that
we want to trace out. This is usefull for example if the system is known to satisfy an area law,
then we can just compute the entanglement entropy for one subsystem of size N//n, and from this
we can compute the slope of the area law.
For example if we set n_values= np.arange(n_min, n_max+ 1) we have maximum information
on the entropy scaling. This can be used to check if the system satisfy an area law.
But we can diminish the number of n_values for sake of speed. For exampl we may consider
only half of the points , n_values=np.arange(n_min, n_max + 1,2) or just two points n_values=[N//4,N//2]"""
n_values=[n_min,int(1/5*N),int(2/5*N),int(3/5*N),int(4/5*N)]
# n_values=[n_min,int(4/5*N)]

# n_values=np.arange(0,int(3/4*N),3)

# Percentage of data excluded from the linear fit to avoid edge effects.
skip_first_percent = 0
skip_last_percent = 0

l_max=200#l_max is the maximum l in the spherical harmonic expansion of the field.

"""mu is the mass of the field."""
# mu=1/m_pl_GeV# In plack masses
mu=0# In plack masses

#%% PRECISION PARAMETERS
"""Here you can set the precision parameters for the varius integrations that
the code performs."""

"""Particle horizon integration parameters"""
# hor_atol=1e-15
# hor_rtol=1e-12
hor_atol=1e-10
hor_rtol=1e-7

horizon_a_min=1e-15
horizon_a_max=1
horizon_size_at_a_min=0


"""Background integration limits, used for the dark density evolution and Friedmann equation 
integration"""
bkg_a_min=1e-15
bkg_a_max=1

"""Precision parameters for the integral of the fluid equation."""
dark_energy_density_evolution_atol = 1e-10
dark_energy_density_evolution_rtol = 1e-8
dark_energy_density_evolution_max_step = 1e-3

""" Precision parameters for the integral of the Friedmann equation to compute t(a)."""
friedmann_atol = 1e-6
friedmann_rtol = 1e-4
friedmann_max_stepsize = 1e-3

"""Ermakov-like equation integration
Those have a great impact on the code performance. While the cosmolgy is solved
once for all, we have to solve the Ermakov-like equation np.sum(l_max*n_values)*len(times) times.
If the code run slow you can diminish the precison."""
# ermak_atol=1e-10
# ermak_rtol=1e-8
ermak_atol=1e-6
ermak_rtol=1e-4

#%% Debug and warnings
"""Displays warnings and debug"""

"""The higer verbose is, the more warnings the code prints."""
verbose=0

"""For each value of the debug level, we print everything it 
prints with lower debug levels.

if debug_level==0:
    -Print nothing.
if debug_level==1:
    -Print l in the l-cycle. 
if debug_level==2:
    -Plot the bakground cosmology quantities. 
if debug_level==3:
    -Plot some solutions for rho_lj(t). 
if debug_level==4:
    -Plot even more solutions for rho_lj(t).
    -Print the % error on the entanglement entropy."""
debug_level=4

#%% Set the density parameters for the choosen cosmology

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

if params:# It means: if params is not None and not empty
    omega_m0 = params.get("omega_m0", 0.0)
    omega_r0 = params.get("omega_r0", 0.0)
    omega_k0 = params.get("omega_k0", 0.0)
    
    # Calculate omega_l0 if it's not explicitly provided
    omega_l0 = params.get("omega_l0", 1.0 - omega_m0 - omega_r0 + omega_k0)
else:
    raise ValueError(f"Unknown cosmology type: {cosmology}")
    
#%% PARAMETERS DEFINED THROUGH THE PREVIOUS ONES
"""Those settings cannot be changed because the other options
are not yet fully implemented."""

cut_off_types=["fixed_comoving","fixed_physical"]
cut_off_type="fixed_comoving"

horizon_types=["fixed_comoving","fixed_physical","particle_horizon"]
horizon_type="fixed_comoving"

if logspaced_times:
    times=np.round(nm.logspace(t_min, t_max,N_t),4)
else:
    times=np.round(np.linspace(t_min,t_max, N_t),4)
    

    
n_max=n_min+N

#Check if t_ini is smaller then t_min. Raise error otherwise
if t_ini>t_min:
    raise Exception("t_min must be greater then t_ini.")
#Insert t_ini in the times list, as the first entry
times = np.insert(times, 0, t_ini)

"""The important parameters are printed 
when the compute_entanglement_entropy function is called. 
The values of those parameters are also saved in a .txt file in the plot folder."""

important_parameters=["n_min","N","l_max","cut_off","mu","t_min","t_max","N_t",
                      "omega_m0","omega_r0","omega_l0","omega_k0","cut_off_type","horizon_type"]

display_parameter_names={#To display the parameters in the plots
                         "mu":r"$\mu$",
                         "r_sch":r"r_{s}",
                         "l_max":r"l_{\mathrm{max}}",
                         "cut_off":r"b",
                         "t_min":r"$t_{\mathrm{min}}$",
                         "t_max":r"$t_{\mathrm{max}}$",
                         "N_t":r"$N_t$"}


    

            
