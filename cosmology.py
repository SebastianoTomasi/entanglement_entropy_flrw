# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:30:54 2023

@author: Sebastiano Tomasi
"""
import scipy as sp
import numpy as np
from math import sqrt

import simulation_parameters as par
import cosmological_functions as cosm_func

import sys
sys.path.append('C:/Users/sebas/custom_libraries')
import plotting_function as pl
import numerical_methods as nm


"""We distinguish flat and nonflat cases. Even if the formalism is the same,
it is easier to just separate the two."""
["flat","eds","lcdm","rad","ds"]
if par.cosmology=="eds" or par.cosmology=="lcdm" or par.cosmology=="rad":
    #%% GENERAL FLRW COSMOLOGY CASE
    
    
    #Define the dark density evolution. In this way we can manage non standard DE EOS.
    dark_density_evolution_numerical_a=cosm_func.solve_dark_density_evolution_numerical_a()
    scale_parameter_values=dark_density_evolution_numerical_a[0]
    dark_density_evolution_a=sp.interpolate.interp1d(scale_parameter_values, dark_density_evolution_numerical_a[1],
                                        fill_value="extrapolate", assume_sorted=True)

    """Integrate the friedmann equation"""
    friedmann_equation_integrand=cosm_func.create_log_friedmann_equation_integrand(dark_density_evolution_a)
    if __name__ == "__main__" or par.debug_level>=2:#Those are for debugging the cosmology.
        pl.plot([scale_parameter_values,friedmann_equation_integrand(scale_parameter_values)],
                  func_to_compare=lambda y: np.e**(3*y/2),title="Fr. Eq. integrand")
        
    hubble_constant_times_t = nm.integrate(f=friedmann_equation_integrand,
                                              a=np.log(par.bkg_a_min), b=np.log(par.bkg_a_max),
                                              atol=par.friedmann_atol,rtol=par.friedmann_rtol,max_step=par.friedmann_max_stepsize)
    hubble_constant_times_t[0]=np.exp(hubble_constant_times_t[0])#It is y=ln(a)
    scale_parameter_values=hubble_constant_times_t[0]
    if __name__ == "__main__" or par.debug_level>=2:#For debug
        pl.plot(hubble_constant_times_t,title="H0*t",
                func_to_compare=lambda a: 2/3*a**(3/2),#EdS
                  legend=["Numerical","EdS"])
    
    """Compute t(a)==time_a"""
    time=hubble_constant_times_t[1]/par.hubble_constant
    
    unverse_age=time[nm.find_closest_index(scale_parameter_values,1.)]
    if __name__=="__main__":
        print("Age of the universe = ",unverse_age)
    
    """Invert t(a) to obtain a(t). In more detail we have a(t) s.t a(t0)=1 
    where t_0 is the age of the universe"""
    scale_parameter_numerical_t=np.array([time,scale_parameter_values])
    
    if __name__ == "__main__" or par.debug_level>=2:#For debug
        pl.plot(scale_parameter_numerical_t,title="Scale parameter",
                  xlabel=[r"t "],ylabel=[r"a(t)"],legend=["Numerical","EdS"],
                   func_to_compare=lambda t: (3/2*t*par.hubble_constant)**(2/3),
                  # func_to_compare=lambda t:np.exp(par.hubble_constant*t),#dS
                  )#EDS
    
    """Compute the hubble function H(t)=\dot{a}/a and the second derivative of a"""
    scale_factor_t=sp.interpolate.interp1d(time, scale_parameter_values,
                                      fill_value="extrapolate", assume_sorted=True,kind="quadratic")
    cosmic_time_a=sp.interpolate.interp1d( scale_parameter_values,time,
                                      fill_value="extrapolate", assume_sorted=True,kind="quadratic")
    
    h=nm.rms_neigbour_distance(time)#Compute average spacing between time points
    new_lenght=int(2/h)#Use h to obtain the new time array lenght
    time=np.linspace(time[0],time[-1],new_lenght)#Build a new time array with equally spaced points
    scale_parameter_values=scale_factor_t(time)#Compute the corresponding scale parameter values
    
    # print(time[0],time[-1])
    scale_parameter_derivative_numerical_t=nm.derivate(scale_factor_t,time[0],time[-1],points= new_lenght,n=1,order=2)
    
    skip=2#We must skip the first two values because those are wrong
    
    scale_parameter_derivative_numerical_t = (
        scale_parameter_derivative_numerical_t[0][skip:], 
        scale_parameter_derivative_numerical_t[1][skip:]
    )
    if __name__ == "__main__" or par.debug_level>=2:#For debug
        pl.plot(scale_parameter_derivative_numerical_t,
                  ylabel=r"$\dot(a)$",xlabel=r"$t$ ",title="Scale param. derivative",
                   yscale="log",xscale="log",
                   dotted=True, 
                   # xlim=(-0.1,2),ylim=(-0.1,10),
                  legend=["Numerical","EdS"],
                  func_to_compare=lambda t:par.hubble_constant*(3/2*par.hubble_constant*(t+1e-50))**(-1/3))
    
    time=time[skip:]#Need to skip also here to make the dimensions all the same
    scale_parameter_values=scale_parameter_values[skip:]
    hubble_function_numerical_t=np.array([time,scale_parameter_derivative_numerical_t[1]/scale_parameter_values])
    hubble_function_t=sp.interpolate.interp1d(hubble_function_numerical_t[0], hubble_function_numerical_t[1],
                                      fill_value="extrapolate", assume_sorted=True,kind="quadratic")
    if __name__ == "__main__" or par.debug_level>=2:#For debug
        pl.plot([hubble_function_numerical_t[0], hubble_function_numerical_t[1]],title="Hubble function",
                  xlabel=["t "],ylabel=["H(t)"],
                    yscale="log",xscale="log",
                    legend=["Numerical","EdS"],
                   dotted=True, connected_dots=True,
                    # xlim=(1e-3,10), ylim=(1e-1,1e20),
                    # ylim=(-1e3,1e1),
                  func_to_compare=lambda t:2/(3*(t+1e-30)))
    """COMPUTE THE HORIZON ACCORDING TO THE USER CHOICHE"""
    """Ifs on the possible user choiches"""
    if par.horizon_type=="fixed_comoving":
        aux =lambda t: par.H0_c
        comoving_horizon_t=np.vectorize(aux)
    elif par.horizon_type=="fixed_physical":
        def comoving_horizon_t(t):
            a=scale_factor_t(t)
            return par.H0_c/a
    
    elif par.horizon_type=="particle_horizon":
        def rescaled_hubble_function_a(a):
            E=cosm_func.create_rescaled_hubble_function_a(dark_density_evolution_a)
            return E(a)
            
        def horizon_integrand_a(a):
            return 1/(rescaled_hubble_function_a(a)*a**2)
        
        comoving_horizon_numerical_a=nm.integrate(horizon_integrand_a,
                                                    par.horizon_a_min,
                                                    par.horizon_a_max,
                                                    Fa=par.horizon_size_at_a_min,
                                                    rtol=par.hor_rtol,atol=par.hor_atol)
        comoving_horizon_a=sp.interpolate.interp1d(comoving_horizon_numerical_a[0],par.H0_c*comoving_horizon_numerical_a[1],
                                            fill_value="extrapolate", assume_sorted=True)
        
        comoving_horizon_t=sp.interpolate.interp1d(cosmic_time_a(comoving_horizon_numerical_a[0]),par.H0_c*comoving_horizon_numerical_a[1],
                                            fill_value="extrapolate", assume_sorted=True)
    else:
        raise Exception(f"horyzion_type={par.horizon_type} not recognized.")
      
    """Definition of the cut off function, depends on user choiche."""
    if par.cut_off_type=="fixed_comoving":
        def aux(a):
            return par.cut_off
        cut_off_t=np.vectorize(aux)
    elif par.cut_off_type=="fixed_physical":
        def cut_off_t(t):
            a=scale_factor_t(t)
            return par.cut_off/a
    else:
        raise Exception(f"cut_off_type={par.horizon_type} not recognized.")
        
    
    
    if par.horizon_type=="particle_horizon":
        a_for_plot=np.linspace(par.bkg_a_min, par.bkg_a_max,200)
        pl.plot([a_for_plot, comoving_horizon_t(a_for_plot)],
                  dotted=True,
                  title="Comoving particle horizon radius",
                  xlabel=r"a",
                  ylabel=r"$R_H^0$",
                  legend=["Numerical","Exact"],
                    # xscale="log",
                    yscale="log",
                  # xlim=(par.a_min,par.a_max),
                  # ylim=(comoving_horizon_a(par.a_min)*(1-0.05),comoving_horizon_a(par.a_max)*(1+0.05)),
                    # func_to_compare=lambda a: 3*par.H0_c*2/3*a**(1/2)
                    func_to_compare=lambda a: par.H0_c*2*sqrt(a)#EDS
                    # func_to_compare=lambda a: par.H0_c*(-1/a+1/par.horizon_a_min)#De sitter
                  )
        
elif par.cosmology=="ds":
    def scale_factor_tt(t):
        return np.exp(par.hubble_constant*t)
    scale_factor_t=np.vectorize(scale_factor_tt)
    
    def hubble_function_tt(t):
        return par.hubble_constant
    hubble_function_t=np.vectorize(hubble_function_tt)
    
    def H0_c(t):
        return par.H0_c
    comoving_horizon_t=np.vectorize(H0_c)
    def const_cut_off(a):
        return par.cut_off
    cut_off_t=np.vectorize(const_cut_off)
#%%
elif par.cosmology=="snyder":
    
    def friedmann_equation_log_integrand(y):
        a=np.exp(y)
        return -a/np.sqrt(1/a-1)
        
    constants_times_t = nm.integrate(f=friedmann_equation_log_integrand,
                                              a=np.log(par.bkg_a_min), b=np.log(par.bkg_a_max),
                                              atol=par.friedmann_atol,rtol=par.friedmann_rtol,max_step=par.friedmann_max_stepsize)
    constants_times_t[0]=np.exp(constants_times_t[0])#It is y=ln(a)
    scale_parameter_values=constants_times_t[0]
    
    """Compute t(a)==time_a"""
    time=constants_times_t[1]/(par.c*np.sqrt(par.k))
    time=time-time[-1]
    
    collapse_time=time[0]
    if __name__=="__main__":
        print("Collapse time = ", collapse_time)
    
    """Invert t(a) to obtain a(t). In more detail we have a(t) s.t a(t0)=1 
    where t_0 is the age of the universe"""
    
    time=np.flip(time)#They are in reverse, so we just flip everything
    scale_parameter_values=np.flip(scale_parameter_values)
    
    

    
    scale_parameter_numerical_t=np.array([time,scale_parameter_values])
    
    alfa=np.linspace(0, np.pi,200)
    t=(alfa+np.sin(alfa))/(2*par.c*np.sqrt(par.k))
    a=(1+np.cos(alfa))/2
    analytical_collapse=[t,a]
    
    if __name__ == "__main__" or par.debug_level>=2:#For debug
        pl.plot([scale_parameter_numerical_t,analytical_collapse],title="Scale parameter",
                  xlabel=[r"t "],ylabel=[r"a(t)"],legend=["Numerical","Analytical"],
                  )#EDS
        
        
    # Get unique time values and their first indices
    unique_time, inverse_indices = np.unique(time, return_inverse=True)
    # Average the scale_parameter_values for each unique time value
    scale_means = np.array([scale_parameter_values[inverse_indices == i].mean() for i in range(len(unique_time))])
    # Display results for verification
    unique_time, scale_means
    """Compute the hubble function H(t)=\dot{a}/a and the second derivative of a"""
    scale_factor_t=sp.interpolate.interp1d(unique_time, scale_means,
                                      fill_value="extrapolate", assume_sorted=True,kind="quadratic")
    
    # Step 1: Get unique scales and the indices of their first occurrences
    unique_scales, inverse_indices = np.unique(scale_parameter_values, return_inverse=True) 
    # Step 2: Calculate the mean of 'time' values for each unique 'scale' value
    time_means = np.array([time[inverse_indices == i].mean() for i in range(len(unique_scales))])
    # Step 3: Create the interpolation function
    cosmic_time_a = sp.interpolate.interp1d(unique_scales, time_means,
        fill_value="extrapolate", assume_sorted=True, kind="quadratic")
    
    h=nm.rms_neigbour_distance(time)#Compute average spacing between time points
    new_lenght=int(2/h)#Use h to obtain the new time array lenght
    time=np.linspace(time[0],time[-1],new_lenght)#Build a new time array with equally spaced points
    scale_parameter_values=scale_factor_t(time)#Compute the corresponding scale parameter values
    
    # print(time[0],time[-1])
    scale_parameter_derivative_numerical_t=nm.derivate(scale_factor_t,time[0],time[-1],points= new_lenght,n=1,order=2)
    
    skip=2#We must skip the first two values because those are wrong
    
    scale_parameter_derivative_numerical_t = (
        scale_parameter_derivative_numerical_t[0][skip:], 
        scale_parameter_derivative_numerical_t[1][skip:]
    )
    if __name__ == "__main__" or par.debug_level>=6:#For debug
        pl.plot([scale_parameter_derivative_numerical_t[0],-np.array(scale_parameter_derivative_numerical_t[1])],
                  ylabel=r"$-\dot{a}$",xlabel=r"$t$ ",title="Negative scale param. derivative",
                    yscale="log",
                    xscale="log",
                   dotted=True, 
                   # xlim=(-0.1,2),ylim=(-0.1,10),
                  legend=["Numerical","EdS"])
    
    time=time[skip:]#Need to skip also here to make the dimensions all the same
    scale_parameter_values=scale_parameter_values[skip:]
    hubble_function_numerical_t=np.array([time,scale_parameter_derivative_numerical_t[1]/scale_parameter_values])
    hubble_function_t=sp.interpolate.interp1d(hubble_function_numerical_t[0], hubble_function_numerical_t[1],
                                      fill_value="extrapolate", assume_sorted=True,kind="quadratic")
    if __name__ == "__main__" or par.debug_level>=2:#For debug
        pl.plot([hubble_function_numerical_t[0], -hubble_function_numerical_t[1]],title="Negative Hubble function",
                  xlabel=["t "],ylabel=["-H(t)"],
                    yscale="log",xscale="log",
                    legend=["Numerical","EdS"],
                   dotted=True, connected_dots=True,
                    # xlim=(1e-3,10), ylim=(1e-1,1e20),
                    # ylim=(-1e3,1e1),
                    )
        
    """COMPUTE THE HORIZON ACCORDING TO THE USER CHOICHE"""
    """Ifs on the possible user choiches"""
    if par.horizon_type=="fixed_comoving":
        aux =lambda t: par.H0_c
        comoving_horizon_t=np.vectorize(aux)
    elif par.horizon_type=="fixed_physical":
        def comoving_horizon_t(t):
            a=scale_factor_t(t)
            return par.H0_c/a
    
    elif par.horizon_type=="particle_horizon":
        def rescaled_hubble_function_a(a):
            E=cosm_func.create_rescaled_hubble_function_a(dark_density_evolution_a)
            return E(a)
            
        def horizon_integrand_a(a):
            return 1/(rescaled_hubble_function_a(a)*a**2)
        
        comoving_horizon_numerical_a=nm.integrate(horizon_integrand_a,
                                                    par.horizon_a_min,
                                                    par.horizon_a_max,
                                                    Fa=par.horizon_size_at_a_min,
                                                    rtol=par.hor_rtol,atol=par.hor_atol)
        comoving_horizon_a=sp.interpolate.interp1d(comoving_horizon_numerical_a[0],par.H0_c*comoving_horizon_numerical_a[1],
                                            fill_value="extrapolate", assume_sorted=True)
        
        comoving_horizon_t=sp.interpolate.interp1d(cosmic_time_a(comoving_horizon_numerical_a[0]),par.H0_c*comoving_horizon_numerical_a[1],
                                            fill_value="extrapolate", assume_sorted=True)
    else:
        raise Exception(f"horyzion_type={par.horizon_type} not recognized.")
      
    """Definition of the cut off function, depends on user choiche."""
    if par.cut_off_type=="fixed_comoving":
        def aux(a):
            return par.cut_off
        cut_off_t=np.vectorize(aux)
    elif par.cut_off_type=="fixed_physical":
        def cut_off_t(t):
            a=scale_factor_t(t)
            return par.cut_off/a
    else:
        raise Exception(f"cut_off_type={par.horizon_type} not recognized.")
        
    
    
    if par.horizon_type=="particle_horizon":
        a_for_plot=np.linspace(par.bkg_a_min, par.bkg_a_max,200)
        pl.plot([a_for_plot, comoving_horizon_t(a_for_plot)],
                  dotted=True,
                  title="Comoving particle horizon radius",
                  xlabel=r"a",
                  ylabel=r"$R_H^0$",
                  legend=["Numerical","Exact"],
                    # xscale="log",
                    yscale="log",
                  # xlim=(par.a_min,par.a_max),
                  # ylim=(comoving_horizon_a(par.a_min)*(1-0.05),comoving_horizon_a(par.a_max)*(1+0.05)),
                    # func_to_compare=lambda a: 3*par.H0_c*2/3*a**(1/2)
                    func_to_compare=lambda a: par.H0_c*2*sqrt(a)#EDS
                    # func_to_compare=lambda a: par.H0_c*(-1/a+1/par.horizon_a_min)#De sitter
                  )
    
else:
    #%% FLAT CASE
    """We define the trvial functions in the flat spacetime case.
    We vectorize them for sake of generality, so we can use the same code also
    in the flat case."""
    def one(t):
        return 1
    scale_factor_t=np.vectorize(one)
    def zero(t):
        return 0
    hubble_function_t=np.vectorize(zero)
    
    def H0_c(t):
        return par.H0_c
    comoving_horizon_t=np.vectorize(H0_c)
    def const_cut_off(a):
        return par.cut_off
    cut_off_t=np.vectorize(const_cut_off)



#%%




