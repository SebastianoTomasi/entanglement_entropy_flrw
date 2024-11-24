# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:51:39 2023

@author: Sebastiano Tomasi
"""
import time as timee
import scipy as sp
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
import numerical_methods as nm


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

#%% PLOT THE ENTANGLEMENT ENTROPY AS A FUNCTION OF SCHWARZSCHILD TIME INSTEAD OF COMOVING


def dTdt(t):
    """Derivative of the schwarzschild time respect the comoving time."""
    a=cosm.scale_factor_t(t)
    R_b=par.r_b*a
    dR_b_dt=par.c*np.sqrt(par.k*(1/a-1))
    numerator= np.sqrt(
        1 - par.r_s / R_b + (1 / par.c * dR_b_dt)**2
    )
    denominator = 1 - par.r_s / R_b
    
    result = numerator / denominator

    return result



t0=par.collapse_time
a_s=par.r_s/par.r_b

print(par.r_b)
print(par.r_s)
print(t0)
print(a_s)
print(par.k)

print(par.t_rs)

schwarzschild_time_numerical_t=nm.integrate(f=dTdt, a=0, b=par.t_rs,
                       atol=par.friedmann_atol,rtol=par.friedmann_rtol,max_step=par.friedmann_max_stepsize)

#schwarzschild_time_t is the schwarzschild time as a function of the comoving time t.
schwarzschild_time_t=sp.interpolate.interp1d(schwarzschild_time_numerical_t[0], schwarzschild_time_numerical_t[1],
                                    fill_value="extrapolate", assume_sorted=True)

tbars=[float(schwarzschild_time_t(t)) for t in par.times[1:] if t<par.t_rs]

how_many_schw_times=len(tbars)
pl.plot([par.times[1:how_many_schw_times+1],tbars],
        # yscale="log",
        xlabel=r"$t$",ylabel=r"$T$",
        title="Schwarzschild time as a function of comoving time",
        dotted=True)

save_with_name=par.save_plot_dir+"/comoving_ee_slopes_tbar_{fixed_name_right}"

pl.plot([tbars,simulation.comoving_angular_coefficients[:how_many_schw_times]],
        title="Comoving Entanglement entropy slope at various Schwarzschild times.",
        xlabel="$T$ [Gy]",
        ylabel="Slope",
        # x_ticks=[0,par.t_rs,par.collapse_time],
        yscale="log",
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

save_with_name=par.save_plot_dir+f"/comoving_ee_slopes_tbar_{fixed_name_right}"

pl.plot([tbars,simulation.comoving_angular_coefficients[:how_many_schw_times]],
        title="Comoving Entanglement entropy slope at various Schwarzschild times.",
        xlabel="$T$ [Gy]",
        ylabel="Slope",
        # x_ticks=[0,par.t_rs,par.collapse_time],
        # yscale="log",
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)


save_with_name=par.save_plot_dir+f"/physical_ee_slopes_tbar_{fixed_name_right}"

pl.plot([tbars,simulation.physical_angular_coefficients[:how_many_schw_times]],
        title="Physical entanglement entropy slope at various Schwarzschild times.",
        xlabel="$T$ [Gy]",
        ylabel="Slope",
        # xscale="log",
        # yscale="log",
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

# factor=cosm.scale_factor_t(par.times[1:how_many_schw_times+1])
# pl.plot([tbars,simulation.comoving_angular_coefficients[:how_many_schw_times]/factor**2],
#         title="Physical entanglement entropy slope at various Schwarzschild times.",
#         xlabel="$T$ [Gy]",
#         ylabel="Slope",
#         xscale="log",
#         # yscale="log",
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








