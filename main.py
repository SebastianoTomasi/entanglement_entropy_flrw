# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:51:39 2023

@author: Sebastiano Tomasi
"""
import time as timee
import scipy as sp
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
entropy_scale=par.cut_off**(2)

times = par.times[1:]  # Slice par.times from the second element onward



"""Plot the entanglement entropy scaling for some selectd times"""
selected=[simulation.comoving_entanglement_entropy_scaling_t[i] for i in [0, (len(par.times)-1)//2, (len(par.times)-1)-1]]
legend = [f"t={times[i]}" for i in [0, (len(times)-1)//2, len(times)-1]]

save_with_name=f"{par.fixed_name_left}comoving_ee_scaling_t_{par.fixed_name_right}"
pl.plot(selected,
        # title="Entanglement entropy scaling at various times",
        title=None,
        xlabel="Area",
        ylabel="Entropy",
        legend=legend,dotted=True,connected_dots=True,
        save=par.save_plots, name= save_with_name)

"""Plot the fitted angular coefficients and their % errors."""
save_with_name=f"{par.fixed_name_left}comoving_ee_slopes_t_{par.fixed_name_right}"
#scale it. m=S_N/(cut_off*N)^2 then m*b^2=S_N/N^2
pl.plot([times,np.array(simulation.comoving_angular_coefficients)],
        # title="Entanglement entropy best fit slope at various times.",
        title=None,
        xlabel="$t$ ",
        ylabel="Slope",
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

save_with_name=f"{par.fixed_name_left}comoving_ee_slopes_t_tiks{par.fixed_name_right}"
pl.plot([times,simulation.comoving_angular_coefficients],
        # title="Entanglement entropy comoving slope at various times.",
        title=None,
        xlabel="$t$ ",
        ylabel="Comoving Slope",
        x_ticks=[par.t_min,par.t_rs,par.collapse_time],
        x_ticklabels = [
                        r"$t_{\text{min}}=$" + str(f"{par.t_min:.2g}"),
                        r"$t_{\text{rs}}=$" + str(f"{par.t_rs:.2g}"),
                        r"$t_{\text{c}}=$" + str(f"{par.collapse_time:.2g}")
                        ],
        yscale="log",
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

save_with_name=f"{par.fixed_name_left}comoving_ee_slopes_t_errors{par.fixed_name_right}"
pl.plot([times,simulation.comoving_angular_coefficients_errors],
        # title="Fit errors on the angular coefficients",
        title=None,
        xlabel="t",
        ylabel="$\%$ error", legend=[None],
        dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)
"""Ent"""
save_with_name=f"{par.fixed_name_left}physical_ee_slopes_t_{par.fixed_name_right}"
pl.plot([times,cosm.scale_factor_t(times)**(-2)*simulation.comoving_angular_coefficients],
        # title="Entanglement entropy physical slope at various.",
        title=None,
        xlabel="$t$ ",
        ylabel="Physical Slope",
        yscale="log",
        x_ticks=[0,par.t_rs,par.collapse_time],
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)


#%% PLOT THE ENTANGLEMENT ENTROPY AS A FUNCTION OF SCHWARZSCHILD TIME INSTEAD OF COMOVING

def dTdt(t):
    """Derivative of the schwarzschild time respect the comoving time."""
    a=cosm.scale_factor_t(t)
    R_b=par.r_b*a
    dR_b_dt=par.c*np.sqrt(par.k*(1/a-1))
    numerator= np.sqrt(1 - par.r_s / R_b + (1 / par.c * dR_b_dt)**2)
    denominator = 1 - par.r_s / R_b
    result = numerator / denominator
    return result

schwarzschild_time_numerical_t=nm.integrate(f=dTdt, a=0, b=par.t_rs,
                       atol=par.friedmann_atol,rtol=par.friedmann_rtol,max_step=par.friedmann_max_stepsize)

#schwarzschild_time_t is the schwarzschild time as a function of the comoving time t.
schwarzschild_time_t=sp.interpolate.interp1d(schwarzschild_time_numerical_t[0], schwarzschild_time_numerical_t[1],
                                    fill_value="extrapolate", assume_sorted=True)

tbars=[float(schwarzschild_time_t(t)) for t in times if t<par.t_rs]

how_many_schw_times=len(tbars)
pl.plot([par.times[1:how_many_schw_times+1],tbars],
        # title="Schwarzschild time as a function of comoving time",
        title=None,
        # yscale="log",
        xlabel=r"$t$",ylabel=r"$T$",
        dotted=True)

save_with_name=f"{par.fixed_name_left}comoving_ee_slopes_T{par.fixed_name_right}"
pl.plot([tbars,simulation.comoving_angular_coefficients[:how_many_schw_times]],
        # title="Comoving Entanglement entropy slope at various Schwarzschild times.",
        title=None,
        xlabel="$T$ ",
        ylabel="Comoving Slope",
        # x_ticks=[0,par.t_rs,par.collapse_time],
        yscale="log",
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)


save_with_name=f"{par.fixed_name_left}physical_ee_slopes_T{par.fixed_name_right}"
pl.plot([tbars,simulation.comoving_angular_coefficients[:how_many_schw_times]*cosm.scale_factor_t(par.times[1:how_many_schw_times+1])**(-2)],
        # title="Physical entanglement entropy slope at various Schwarzschild times.",
        title=None,
        xlabel="$T$ ",
        ylabel="Physical Slope",
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=[None],dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

# %%


save_with_name=f"{par.fixed_name_left}sigmaR_lj{par.fixed_name_right}"
auxilliary=[ [elem[0],np.real(elem[1])] for  elem in simulation.sigma_l_t_for_plot[0]]
pl.plot(auxilliary,
        # title="Physical entanglement entropy slope at various Schwarzschild times.",
        title=None,
        # xscale="log",
        yscale="log",
        xlabel="$t$ ",
        ylabel=r"Re$(\Sigma_{j}^{l})$",
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=simulation.sigma_l_t_for_plot[1],
        # dotted=True,
        save=par.save_plots,name=save_with_name)
# %%

save_with_name=f"{par.fixed_name_left}sigmaI_lj{par.fixed_name_right}"
auxilliary=[ [elem[0],np.imag(elem[1])] for  elem in simulation.sigma_l_t_for_plot[0]]
pl.plot(auxilliary,
        # title="Physical entanglement entropy slope at various Schwarzschild times.",
        title=None,
        # xscale="log",
        # yscale="log",
        xlabel="$t$ ",
        ylabel=r"Im$(\Sigma_{j}^{l})$",
        x_ticklabels=[str(label) for label in list(np.round(np.array([0,par.t_rs,par.collapse_time]),3))],
        legend=simulation.sigma_l_t_for_plot[1],
        # dotted=True,connected_dots=True,
        save=par.save_plots,name=save_with_name)

#%%
"""Save data."""
if not par.plot_saved_data:
    if par.save_data:
        simulation.save_data()
        
minutes, seconds = divmod(timee.time() - start_time, 60)
print("\nTime taken to complete the execution: {:.0f} minutes {:.0f} seconds".format(minutes, seconds))








