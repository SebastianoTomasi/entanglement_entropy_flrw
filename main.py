# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:51:39 2023

@author: Sebastiano Tomasi
"""
import time as timee
from datetime import datetime
import numpy as np
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
"""Plot the entanglement entropy"""
par.save_plots=True
legend=[]
for i in range(1,len(par.times),4):
    legend.append("t="+str(par.times[i])+r" $Gy$")

save_with_name="ee_scaling_t_"+par.cosmology+"_"+it_datetime
pl.plot(simulation.entanglement_entropy_scaling_t,
        title="Entanglement entropy scaling at various times",
        xlabel="Area",
        ylabel="Entropy",
        legend=legend,dotted=True,connected_dots=True,
        save=par.save_plots, name= save_with_name)

"""Plot the fitted angular coefficients and their % errors."""
save_with_name="ee_slopes_t_"+par.cosmology+"_"+it_datetime
pl.plot([par.times[1:],simulation.angular_coefficients],
        title="Entanglement entropy slope at various times.",
        xlabel="$t$ [Gy]",
        ylabel="Slope",
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

save_with_name="ee_slopes_t_0_"+par.cosmology+"_"+it_datetime
pl.plot([par.times[1:],simulation.angular_coefficients],
        title="Entanglement entropy slope at various times.",
        xlabel="$t$ [Gy]",
        ylabel="Slope",
        x_ticks=[0,par.t_rs,par.collapse_time],
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

save_with_name="ee_slopes_t_errors"+par.cosmology+"_"+it_datetime
pl.plot([par.times[1:],simulation.angular_coefficients_errors],
        title="Percentage errors on the angular coefficients",
        xlabel="Time",
        ylabel="$\%$ error", legend=[None],
        dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)
print(f"The time at which the radius becomes equal to the Schwarzshild radius is \n t = {par.t_rs}")

# index = 1
# # Ensure entanglement_entropy_scaling_t is a NumPy array
# entanglement_entropy_scaling_t = np.array(simulation.entanglement_entropy_scaling_t)

# # Extract the required values
# entanglement_entropy_t = [simulation.times, entanglement_entropy_scaling_t[:, 1, index]]
# save_with_name="ee_t_fix_shell"+par.cosmology+"_"+it_datetime
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








