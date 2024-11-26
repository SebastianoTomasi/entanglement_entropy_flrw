# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:47:00 2023

@author: Sebastiano Tomasi
"""
import scipy as sp
import numpy as np
from numpy import sqrt,log

import simulation_parameters as par
import cosmology as cosm
import coupling_matrix as cm

import sys
sys.path.append('C:/Users/sebas/Documents/GitHub/custom_libraries')
import plotting_function as pl



#%%
"""We first define some functions that we will need in the main cycle."""

def compute_entanglement_entropy(xi):
    """This function takes as input xi, which is a vector containing all xi_i, as
    defined in the paper. It returns the entanglement entropy associated to those."""
    number_of_negative_xi=0
    s=0
    for val in xi:
        if par.verbose>1 and val.imag>1e-10:
            print(val)
        if val>0:
            s+=-log(1-val)-val/(1-val)*log(val)
        elif val<0:
            # print(val)
            if abs(val)>1e-30:
                number_of_negative_xi+=1
                if par.verbose>=1:
                    print(f"Xi is negative! xi={val}<0")
                pass
    if par.verbose>=1 and number_of_negative_xi>0:
        print(f"Number of negative xi ={number_of_negative_xi}!")
    return s

def M_t(t):
    """This is the effective mass of one oscillator
     when we interpret the discretized Hamiltonian H_{lm} as
     the Hamiltonian for a system of harmonic oscillators. (See the paper.)"""
    return cosm.scale_factor_t(t)**3/par.c**2

#%%

def ermakov_like_equation(t,y,omega_lj2):
    """Takes in input
    t: cosmological time
    y: vector containing the initial conditions on rho and 
        its derivative y[0]=rho(t_ini), y[1]=rho'(t_ini)
    omega_lj2: callable, the frequency as a function of time.
    
    This function is the first order system which is equivalent
    to the Ermakov-like equation.
    """
    
    fun=[0]*2
    fun[0] = y[1]
    fun[1] = -3*cosm.hubble_function_t(t)*y[1]-omega_lj2(t)*y[0]+par.c**4/(cosm.scale_factor_t(t)**6*y[0]**3)
    return np.array(fun)

def solve_ermakov_like(t_min, t_max, rho_t_min, drho_t_min, omega_lj2, full_solution=True):
    """
    Takes in input:
    t_min: starting point of the integration
    t_max: ending point of the integration
    rho_t_min, drho_t_min: initial conditions
    
    omega_lj2: callable, the frequency as a function of time.
    full_solution: if True, return the full solution from t_min to t_max (default is False).

    Returns:
        If full_solution is False: (rho_t_max, drho_t_max): solution and its derivative, computed at the ending point of the integration.
        If full_solution is True (default): [t, rho_t]: full solution from t_min to t_max.
    """

    init_cond = [rho_t_min, drho_t_min]  # Initial conditions for the system, passed to scipy solver.
    
    # Perform the integration
    rk4_result = sp.integrate.solve_ivp(
        ermakov_like_equation, t_span=(t_min, t_max), y0=init_cond, 
        method="RK45", atol=par.ermak_atol, rtol=par.ermak_rtol, 
        args=(omega_lj2,)
    )
    
    # If full_solution is True, return the full solution
    if full_solution:
        rho_t = rk4_result.y[0]
        t = rk4_result.t
        return [t, rho_t]
    
    # Otherwise, return the values at t_max
    rho_t_max = rk4_result.y[0][-1]
    drho_t_max = rk4_result.y[1][-1]
    return [rho_t_max, drho_t_max]

def approx_rho_eds(t,C1,C2, gamma_lj):
    """
    Computes the approximated expression for rho_lj(t) for the frequency plj->0.
    

    Parameters:
    t (float): Time variable.
    C1 (float): Constant C1.
    c (float): Constant c.
    t0 (float): Constant t0.
    Gamma_lj (float): Constant Gamma_lj.

    Returns:
    float: The computed rho value.
    """

    plj = 27 * (par.c ** 3) * (cosm.unverse_age ** 2) * (gamma_lj ** 1.5)

    
    # Calculate rho
    rho = t**(1/3)*(-C1+9*(1+C2)*plj**(4/3)*t)
    
    return rho

#%%

def run():
    """ For each time vale, we have to:
        1)  Define the time dependent coupling matrix,
            coupling_matrix_lt, (\tilde{\boldsymbol{C}} in the paper). 
        2)  Compute the eigenvalues of the time independent part of the coupling_matrix_lt,
            which we call coupling_matrix_l, which are called gamma2_lj.
            With those we can define the frequencies
            omega_lj2=c/a(t)sqrt(gamma_lj2+(mu*a(t)*b)), where a(t) is the scale
            factor, mu the field mass, b the cut-off value, c the speed of light.
        3)  Solve the Ermakov-like equation from the initial time t_ini at which
            we impose initia conditions, up to times[i]. To make this step more 
            efficient we can use the solution rho(times[i]) as initial condition
            for the next time step. In this way we do not compute the same thing
            over and over.
        4)  Define the covariance matrix sigma_l_t trough the solution of the Ermakov-like 
            equation.
        5)  Compute the entanglement entropy scaling of sigma_l_t.
        6)  Perform the summation S(l_max)=\sum_{l=0}^{l_max} (2*l+1)S(sigma_l_t). To improve performance
            we save the value S(l_max) and check convergence by computing the percentage
            difference between S(l_max) and S(l_max-1). If it is under the choosen tolerance,
            save the result.
            
        We assume that the cut-off is a constant comoving value.
        
            """
    
    """Initialize the vectors used for containing the final results
    for the entropy as a function of time
    comving_entanglement_entropy_scaling_t=[[area,EntEntr at times[1]],[area,EntEntr at times[2]]...]"""
    len_n_values=len(par.n_values)
    comving_entanglement_entropy_scaling_t=[]
    physical_entanglement_entropy_scaling_t=[]
    
    """Since the initial conditions are on rho_lj, we have to save the values
    for each lj pair. So at each time cycle, we recylce the solution obtained 
    at the previos time. By doing it this way we loose the ability to make
    a smart alghoritm that stops when a desired tolerance on the entropy is achieved.
    But we gain in simplicity of the alghoritm."""
    
    rho_t_prec=np.zeros((par.l_max+1,par.N ), dtype=float)#l_max+1 rows and N columns.
    drho_t_prec=np.zeros((par.l_max+1,par.N ), dtype=float)
    
    if par.debug_level>=3:
        rho_for_plot_t,rho_for_plot_legend=[],[]
        
    for i in range(1,len(par.times)):#Starts from one because of the definition of times.
        """Initialize the vectors used for containing the partial results for the 
        entanglement entropy, entropy_nl, and the final result, entropy_n. We have
        to define those here since when the time steps forward we have to reset the 
        values to zero."""
        entropy_nl=np.zeros(len_n_values)
        entropy_n=np.zeros(len_n_values)
        
        """Errors will contain the maximum percentage difference
        between the entropy computed at l_max-1 and l_max. It is
        a clear quantifier of the precision of the final result."""
        max_errors=[]
        
        for l in range(0,par.l_max+1):
            if par.debug_level>=1:
                if l%100==0:
                    if l==0:
                        print(f"l = {l}, t = {par.times[i]}")
                    else:
                        print(f"l = {l}")
                        
            
            """We have to define the time dependent coupling matrix coupling_matrix_lt,
            which obviously depends on time, but also on the spherical harmonic index l.
            We generate directly the time independent part separately from the time dependent one.
            So we define coupling_matrix_l as the time independent part (\boldsymbol{C} in the paper)."""
            
            """If you want to be able to use a time dependent horizon and cut off, then
            here we should write someting like 
            par.N=int(cosm.comoving_horizon_t(times[i])/cosm.cut_off_t(times[i])).
            This feature is not implemented yet."""
            
            coupling_matrix_l=cm.generate_ticm(l)
            # print(coupling_matrix_l)
            
            """Now we have to compute the eigenvalues, gamma_l2 and the
            orthonormal basis change that diagonalizes coupling_matrix_l, which we call U as in the paper,
            such that \Gamma^2=U*C*U^T. We will call gamma_lj2 the j-th element of gamma_l2."""
            gamma_l2, UT =sp.linalg.eigh(coupling_matrix_l)
            U=np.transpose(UT)
            # print(gamma_l2)
          
            
            
            """Now we have to solve the Ermakov-like equation for each time independent eigevalue contained gamma_l2.
            We thus perform a cycle on those eigenvalues. The goal is to define the covariance matric \boldsymbol{\Sigma}
            computed at time par.times[i]=t."""
            
            
            
            diagonal_sigma_l_t=[]#Define the diagonal of the covariance matrix here, then we fill it in the next cycle.
            for j,gamma_lj2 in enumerate(gamma_l2):# j gores from 0 to par.N-1 
                
                """We can now define the frequencies appearing in the Ermakov-like equation"""
                def omega_lj2(t):
                    """Takes in input the cosmological time x and returns omega_lj2, (\omega_{lj}^2 in the paper),
                    which is the frequency appearing in the Ermakov-like equation.
                    It needs to be defined globally gamma_l2 and the cosmologcal functions defined in the 
                    cosmology module.
                    """
                    return par.c**2/cosm.scale_factor_t(t)**2*(gamma_lj2+(par.mu*cosm.scale_factor_t(t)*par.cut_off))
                
                """In the first iteration we have to impose the flat space harmonic oscillator initial conditions"""
                if i==1:
                    rho_t_prec[l][j]=1/np.sqrt(M_t(par.t_ini)*sqrt(omega_lj2(par.t_ini)))
                    # print("init_cond:",1/np.sqrt(M_t(par.t_ini)*sqrt(omega_lj2(par.t_ini))))
                    # print("M_t:",M_t(par.t_ini))
                    # print("omega_lj2:",omega_lj2(par.t_ini))
                    # rho_t_prec[l][j]=1
                    # drho_t_prec[l][j]=0. do not need this since it's already zero.
            
                """We then solve the Ermakov-like equation. We obtain the solution
                at time par.times[i]. With this we can construct the time dependent covariance matrix."""
                rho_lj_t,drho_lj_t=solve_ermakov_like(t_min=par.times[i-1],t_max=par.times[i],
                                                          rho_t_min=rho_t_prec[l][j],drho_t_min=drho_t_prec[l][j],
                                                          omega_lj2=omega_lj2,
                                                          full_solution=False)
                
                """For checking the results, we plot some of the solutions of the Ermakov-like equation.
                We solve the full equation from par.t_ini to max(par.times) with standard flat spacetime
                harmonic oscillator initial condtions."""
                if par.debug_level>=4:
                    if i==1:
                        if l==0 or l==par.l_max//2 or l==par.l_max:
                            if j==0 or j==par.N//2 or j==par.N-1:
                                rho_for_plot_t.append(solve_ermakov_like(t_min=par.t_ini,t_max=max(par.times),
                                                                          rho_t_min=1/np.sqrt(M_t(par.t_ini)*sqrt(omega_lj2(par.t_ini)))
                                                                          ,drho_t_min=0.,
                                                                          omega_lj2=omega_lj2,
                                                                          full_solution=True))
                                rho_for_plot_legend.append(f"l={l}, j={j}")
                elif par.debug_level==3:
                    if i==1:
                        if l==0 or l==par.l_max//2:
                            if j==0 or j==par.N//2:
                                rho_for_plot_t.append(solve_ermakov_like(t_min=par.t_ini,t_max=max(par.times),
                                                                          rho_t_min=1/np.sqrt(M_t(par.t_ini)*sqrt(omega_lj2(par.t_ini)))
                                                                          ,drho_t_min=0.,
                                                                          omega_lj2=omega_lj2,
                                                                          full_solution=True))
                                rho_for_plot_legend.append(f"l={l}, j={j}")                            
                                    

                """Save the values to reuse tham in the next cylce on i."""
                rho_t_prec[l][j],drho_t_prec[l][j]=rho_lj_t,drho_lj_t
                
                """We construct the covariance matrix computed at par.times[i]"""
                diagonal_sigma_l_t.append(1/par.hbar*( 1/rho_lj_t**2-1j*M_t(par.times[i])*drho_lj_t/(rho_lj_t) ) )
            
            
            """After the cycle is compleated, we can construct the actual covariance matrix sigma_l_t and
            transform it from the normal coordinates to the physical spatial coordinates."""
            sigma_l_t = np.diag(diagonal_sigma_l_t)
            omega_l_t=np.dot(np.dot(UT,sigma_l_t),U)
                
            for k,n in enumerate(par.n_values):
                if n==par.n_min or n==par.n_max:
                    """If  we consider the complete system or no system
                    the entanglement entropy must be zero."""
                    pass
                else:
                    """Here we need to implement the procedure to compute the entanglement
                    entropy of a complex covariance matrix. We start from its expression in 
                    spatial coordinates: omega_l_t."""
                    
                    """First we define the blocks of omega_l_t"""
                    A,B,B_T,D=cm.slice_matrix(omega_l_t,n)
                    
                    A_real=np.real(A)
                    A_real_inverse=sp.linalg.inv(A_real)
                    
                    
                    B_transpose=B.T
                    B_dagger=B_transpose.conj()
                    
                    A_real_inverse_times_B=A_real_inverse @ B
                    
                    beta=0.5*B_dagger @ A_real_inverse_times_B
                    gamma= np.real(D -0.5* B_transpose @ A_real_inverse_times_B)
                    
                    """Define the matrices for the generalized eigenvalue problem"""
                    dim=int(par.N-n)
                    F=np.block([[-2*gamma   , beta.T            ],
                                [np.eye(dim), np.zeros((dim,dim)) ]])
                    
                    G=np.block([[-beta              , np.zeros((dim,dim)) ],
                                [np.zeros((dim,dim))  , np.eye(dim)   ]])
                    
                    """Solve the generalized eigenvalue problem Fw=\lambda G w"""
                    eigenvalues,_=sp.linalg.eig(F,G)#Can't use eigh sice those are not Hermitian.
                    
                    # We remove infinite eigenvalues
                    num_infinite = np.sum(~np.isfinite(eigenvalues))  # Count how many are infinite, tilde is not operator.
                    if par.verbose>1 and num_infinite > 0:
                        print(f"Warning: There are {num_infinite} infinite eigenvalues.")
                    finite_eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
                    
                    #Take the real part of the eigenvalues, since those should already be real. 
                    imaginary_parts = np.imag(eigenvalues)
                    if par.verbose>=2 and np.any(np.imag(eigenvalues) != 0):  # Check if there are non-zero imaginary parts
                        max_imaginary_part = np.max(np.abs(imaginary_parts))
                        print(f"Warning: The maximum imaginary part being discarded is {max_imaginary_part}.")
                    real_parts = np.real(finite_eigenvalues)
                    
                    # We select only eigenvalues less than 1
                    filtered_eigenvalues = real_parts[real_parts < 1]
                    # print("The eigenvalues are: ", len(filtered_eigenvalues),filtered_eigenvalues)
                    
                    """The filtered eigenvalues are what are usually called the xi values. We 
                    use those to compute the entanglement entropy."""
                    entropy_nl[k]=compute_entanglement_entropy(filtered_eigenvalues)
                    
            """We perform a quality check at the end, i.e. when we arrive at l_max and
            we print the maximum percentage deviation between the entropy in the last step and the 
            previus one."""
            if l==par.l_max:
                nonzero_values= entropy_nl!=0
                nonzero_entropy_nl=entropy_nl[nonzero_values]
                nonzero_entropy_n=entropy_n[nonzero_values]
                #max_error defined as |S(l_max)-S(l_max-1)|/(S(l_max)+S(l_max-1))/2
                max_error=np.max(1 / (0.5 + nonzero_entropy_n / ((2 * l + 1) * nonzero_entropy_nl)))
                max_errors.append(max_error)
                if par.debug_level>=4:
                    print("Maximum relative deviation = {:.4e}".format(max_error))
                print("")
               
                    
            """The entropy_nl must then be summed. When the l cycle ends, entropy_n
            has inside the values of the entropy."""   
            entropy_n=entropy_n+(2*l+1)*entropy_nl
            
        """We then associate each entropy scaling with the time at which it has been computed. 
        We also rescale n, which in this case represent the discretized radial varialbe."""
        comoving_area=(par.cut_off*np.array(par.n_values))**2#*4*np.pi
        physical_area=cosm.scale_factor_t(par.times[i])**2*comoving_area#The physical area is just multiplied by the scale facotr squared.
        
        physical_entanglement_entropy_scaling_t.append([physical_area,entropy_n])
        comving_entanglement_entropy_scaling_t.append([comoving_area,entropy_n])
        
    if par.debug_level>=3:
        pl.plot(rho_for_plot_t,legend=rho_for_plot_legend,title="Some more solutions to the Ermakov equation",
                xlabel=r"$t$",ylabel=r"$\rho_{lj}(t)$",
                xscale="log",
                yscale="log",
                save=par.save_plots,name=f"{par.fixed_name_left}more_rho_lj_t{par.fixed_name_right}")
        pl.plot(rho_for_plot_t,legend=rho_for_plot_legend,title="Some more solutions to the Ermakov equation",
                xlabel=r"$t$",ylabel=r"$\rho_{lj}(t)$",
                # xscale="log",
                # yscale="log",
                save=par.save_plots,name=f"{par.fixed_name_left}more_linear_rho_lj_t{par.fixed_name_right}")
    if par.debug_level==2:
        pl.plot(rho_for_plot_t,legend=rho_for_plot_legend,title="Some solutions to the Ermakov equation",
                xlabel=r"$t$",ylabel=r"$\rho_{lj}(t)$",
                xscale="log",
                yscale="log",
                save=par.save_plots,name=f"{par.fixed_name_left}rho_lj_t{par.fixed_name_right}")
        pl.plot(rho_for_plot_t,legend=rho_for_plot_legend,title="Some more solutions to the Ermakov equation",
                xlabel=r"$t$",ylabel=r"$\rho_{lj}(t)$",
                # xscale="log",
                # yscale="log",
                save=par.save_plots,name=f"{par.fixed_name_left}linear_rho_lj_t{par.fixed_name_right}")
        
    return (comving_entanglement_entropy_scaling_t,physical_entanglement_entropy_scaling_t,max_errors)


