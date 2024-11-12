# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:51:39 2023

@author: Sebastiano Tomasi
"""
import time as timee
import os
from datetime import datetime
import numpy as np
import cosmology as cosm
now = datetime.now()
# Format the date and time in Italian convention: day/month/year hour:minute:second
it_datetime = now.strftime("%d_%m_%Y_%H_%M_%S_")

import simulation_parameters as par
import entanglement_entropy_simulation as sim

import sys
sys.path.append('C:/Users/sebas/Documents/GitHub/custom_libraries')
import plotting_function as pl


start_time = timee.time()

#%%
"""Createa simulation object. It is just a way to neatly
condense the resulting data and some methods."""

simulation=sim.EntanglementEntropySimulation()

if not par.plot_saved_data:
    simulation.run()
    simulation.perform_linear_fit(suppress_report=False)
else:
    simulation.load()

#%%

fixed_name_left=f"{par.save_plot_dir}/{par.cosmology}/"
fixed_name_right=f"{it_datetime}"
os.makedirs(fixed_name_left, exist_ok=True)

"""Plot the entanglement entropy"""
par.save_plots=True
legend=[]
for i in range(1,len(par.times),4):
    legend.append("t="+str(par.times[i])+r" $Gy$")
    

# save_with_name=f"{fixed_name_left}comoving_ee_scaling_t_{fixed_name_right}"
# pl.plot(simulation.comoving_entanglement_entropy_scaling_t,
#         title="Entanglement entropy scaling at various times",
#         xlabel="Area",
#         ylabel="Entropy",
#         legend=legend,dotted=True,connected_dots=True,
#         save=par.save_plots, name= save_with_name)

"""Plot the fitted angular coefficients and their % errors."""
save_with_name=f"{fixed_name_left}comoving_ee_slopes_t_{fixed_name_right}"
pl.plot([par.times[1:],simulation.comoving_angular_coefficients],
        title="Entanglement entropy slope at various times.",
        xlabel="$t$ [Gy]",
        ylabel="Slope",
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

save_with_name=f"{fixed_name_left}comoving_ee_slopes_t_schwtime_{fixed_name_right}"
pl.plot([par.times[1:],simulation.comoving_angular_coefficients],
        title="Entanglement entropy slope at various times.",
        xlabel="$t$ [Gy]",
        ylabel="Slope",
        x_ticks=[0,par.t_rs,par.collapse_time],
        yscale="log",
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

save_with_name=f"{fixed_name_left}comoving_ee_slopes_t_errors{fixed_name_right}"
pl.plot([par.times[1:],simulation.comoving_angular_coefficients_errors],
        title="Percentage errors on the angular coefficients",
        xlabel="Time",
        ylabel="$\%$ error", legend=[None],
        dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

save_with_name=f"{fixed_name_left}physical_ee_slopes_t_{fixed_name_right}"
pl.plot([par.times[1:],simulation.physical_angular_coefficients],
        title="Entanglement entropy physical slope at various.",
        xlabel="$t$ [Gy]",
        ylabel="Physical Slope",
        yscale="log",
        x_ticks=[0,par.t_rs,par.collapse_time],
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)
print(f"The time at which the radius becomes equal to the Schwarzshild radius is \n t = {par.t_rs}")

#%%
    
   
# def tbar(t):
#     R0=par.bh_R0
#     k=par.k
#     a=cosm.scale_factor_t(t)
#     # Check for valid inputs to avoid division by zero or square roots of negative numbers
#     if a == 1 or k == 0 or k * R0**2 >= 1:
#         raise ValueError("Invalid input: check the values of 'a', 'R0', and 'k' to avoid division by zero or imaginary numbers.")
    
#     # Main calculation
#     part1 = np.sqrt((1 - R0**2) / k)
#     term1 = -((-1 + a) * np.sqrt(-a / (-1 + a)))
#     term2 = np.pi * (1/2 + k * R0**2 - (k**(3/2) * R0**3) / np.sqrt(-1 + k * R0**2))
#     term3 = -(1 + 2 * k * R0**2) * np.arctan(np.sqrt(-a / (-1 + a)))
#     term4 = (2 * k**(3/2) * R0**3 * np.arctan((np.sqrt(a / (k - a * k)) * np.sqrt(-1 + k * R0**2)) / R0)) / np.sqrt(-1 + k * R0**2)
    
#     return part1 * (term1 + term2 + term3 + term4)



# save_with_name=par.save_plot_dir+"/comoving_ee_slopes_tbar_{fixed_name_right}"
# t0=par.collapse_time
# a_s=par.r_s/par.bh_R0
# t_s=t0*a_s**(3/2)
# print(par.bh_R0)
# print(par.r_s)
# print(t0)

# print(a_s)
# print(par.k)

# print(t_s)
# tbars=[tbar(t) for t in par.times[1:] if t<t_s]
# pl.plot([tbars,simulation.comoving_angular_coefficients],
#         title="Entanglement entropy slope at various times.",
#         xlabel="$t$ [Gy]",
#         ylabel="Slope",
#         x_ticks=[0,par.t_rs,par.collapse_time],
#         yscale="log",
#         x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
#         legend=[None],dotted=True,connected_dots=True,
#         save=par.save_plots,name=save_with_name)




# index = 1
# # Ensure entanglement_entropy_scaling_t is a NumPy array
# entanglement_entropy_scaling_t = np.array(simulation.entanglement_entropy_scaling_t)

# # Extract the required values
# entanglement_entropy_t = [simulation.times, entanglement_entropy_scaling_t[:, 1, index]]
# save_with_name="ee_t_fix_shell{fixed_name_right}"
# pl.plot(entanglement_entropy_t,
#         title=f"Entanglement entropy for n = {par.n_values[index]} shells as a function of time",
#         xlabel="Time",
#         ylabel="Entropy", legend=[None],
#         dotted=True,connected_dots=True,
#         save=par.save_plots,name=save_with_name)

#%%
"""Save data."""
if not par.plot_saved_data:
    if par.save_data:
        simulation.save_data()
        
minutes, seconds = divmod(timee.time() - start_time, 60)
print("\nTime taken to complete the execution: {:.0f} minutes {:.0f} seconds".format(minutes, seconds))








