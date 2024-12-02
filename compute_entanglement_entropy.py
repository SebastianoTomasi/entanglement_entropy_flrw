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
import numerical_methods as nm
"""
Todos:
    -The code could be made quite a lot faster by saving the results between 
    an iteration on an n_value and the other
"""


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
        If full_solution is True (default): [t, rho_t,drho_t]: full solution from t_min to t_max.
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
        drho_t = rk4_result.y[1]
        rho_t = rk4_result.y[0]
        t = rk4_result.t
        return [t, rho_t,drho_t]
    
    # Otherwise, return the values at t_max
    
    rho_t_max = rk4_result.y[0][-1]
    drho_t_max = rk4_result.y[1][-1]
    return [rho_t_max, drho_t_max]



#%%

def run():
    """ For each time value, we have to:
        1)  Define the time dependent coupling matrix,
            coupling_matrix_lt, (\tilde{\boldsymbol{C}} in the paper). 
        2)  Compute the eigenvalues of the time independent part of the coupling_matrix_lt,
            which we call coupling_matrix_l, which are called gamma_lj2.
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
    comving_entanglement_entropy_scaling_t=[[number_area,EntEntr at times[1]],[number_area,EntEntr at times[2]]...]"""
    len_n_values=len(par.n_values)
    comving_entanglement_entropy_scaling_t=[]
    
    """Since the initial conditions are on rho_lj, we have to save the values
    for each lj pair. So for each time (i), we reuse the solution obtained 
    at the previos time (i-1). By doing it this way we loose the ability to make
    a smart alghoritm that stops when a desired tolerance on the entropy is achieved.
    But we gain in simplicity."""
    
    rho_t_prec=np.zeros((par.l_max+1,par.N ), dtype=float)#l_max+1 rows and N columns.
    drho_t_prec=np.zeros((par.l_max+1,par.N ), dtype=float)
        
    if par.debug_level>=3:#Save the grund state matrix if debug_level is high enough: \boldsymbol{\Sigma}^l(t) in the paper
        sigma_l_t_for_plot,sigma_l_t_for_plot_legend=[],[]
        
    for i in range(1,len(par.times)):#Starts from one because of the definition of times.
        t=par.times[i]
        """Initialize the vectors used for containing the partial results for the 
        entanglement entropy, entropy_nl, and the final result, entropy_n. We have
        to define those here since when the time steps forward we have to reset them to zero."""
        
        entropy_nl=np.zeros(len_n_values)
        entropy_n=np.zeros(len_n_values)
        
        """max_errors will contain the maximum percentage difference
        between the entropy computed at l_max-1 and l_max. It is
        a clear quantifier of the tolerance of the final result."""
        max_errors=[]
        
        for l in range(0,par.l_max+1):#Cycle on the angular momentum index
            if par.debug_level>=1:
                if l%100==0:
                    if l==0:
                        print(f"l = {l}, t = {t}")
                    else:
                        print(f"l = {l}")
                        
            
            """We have to define the time dependent coupling matrix coupling_matrix_lt,
            which obviously depends on time, but also on the spherical harmonic index l.
            We generate directly the time independent part separately from the time dependent one.
            So we define coupling_matrix_l as the time independent part (\boldsymbol{C} in the paper)."""
            
            """Todos: 
                If you want to be able to use a time dependent horizon and cut off, then
                here we should write someting like 
                par.N=int(cosm.comoving_horizon_t(times[i])/cosm.cut_off_t(times[i])).
                This feature is not implemented yet. """
            
            coupling_matrix_l=cm.generate_ticm(l)
            
            """Now we have to compute the eigenvalues, gamma_l2 and the
            orthonormal basis change that diagonalizes coupling_matrix_l, which we call U as in the paper,
            such that \Gamma^2=U*C*U^T. We will call gamma_lj2 the j-th element of gamma_l2.
            
            The maximum value of gamma_lj^2 is around l. If you want to consider higer
            values of l, you should also improve the precision parameters for the 
            ermakov like equation, as the numerical methods always fail for high enough 
            frequencies. Fortunately, the higher the l value the smaller is the contribution."""
            
            gamma_l2, UT =sp.linalg.eigh(coupling_matrix_l)
            U=np.transpose(UT)
            
            # if l%100==0:
            #     print(np.round(U@coupling_matrix_l@UT,4))
            
            """Now we have to solve the Ermakov-like equation for each time independent eigevalue contained gamma_l2.
            We thus perform a cycle on those eigenvalues. The goal is to define the covariance matric \boldsymbol{\Sigma}
            computed at time t=t."""
            
            
            
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
                    # drho_t_prec[l][j]=0 do not need this since it's already zero.
            
                """We then solve the Ermakov-like equation. We obtain the solution
                at time t. With this we can construct the time dependent covariance matrix."""
                rho_lj_t,drho_lj_t=solve_ermakov_like(t_min=par.times[i-1],t_max=t,
                                                          rho_t_min=rho_t_prec[l][j],drho_t_min=drho_t_prec[l][j],
                                                          omega_lj2=omega_lj2,
                                                          full_solution=False)
                
                if np.sign(rho_lj_t)!=np.sign(rho_t_prec[l][j]):
                    """From the general integral solution we know that rho cannot change sign,
                    but if the frequency is very hig, (gamma_lj is hig) then the rk4 method 
                    fail to keep the sign the same. We can adjust this with those operation.
                    If the tolerances in the rk4 solver are tight enough, the alghorithm should
                    never come in here, but if not, this will significantly improve results."""
                    rho_lj_t*=-1
                    drho_lj_t*=-1
                    if par.verbose>=2:
                        print(f"Sign change in rho detected and corrected! \nOccured with: \ngamma_lj2={gamma_lj2}\nt={t}")
                
                """For checking the results, we plot some of the solutions of the Ermakov-like equation.
                We solve the full equation from par.t_ini to max(par.times) with standard flat spacetime
                harmonic oscillator initial condtions."""
                if par.debug_level>=3:
                    if i==1:
                        if l==0 or l==par.l_max//2 or l==par.l_max:
                            if j==0 or j==par.N//2 or j==par.N-1:
                                t_plot,rho_lj_t_plot,drho_lj_t_plot=solve_ermakov_like(t_min=par.t_ini,t_max=max(par.times),
                                                                          rho_t_min=1/np.sqrt(M_t(par.t_ini)*sqrt(omega_lj2(par.t_ini)))
                                                                          ,drho_t_min=0.,
                                                                          omega_lj2=omega_lj2,
                                                                          full_solution=True)
                                sigma_l_t_for_plot.append([t_plot,1/par.hbar*( 1/rho_lj_t_plot**2-1j*M_t(t_plot)*drho_lj_t_plot/(rho_lj_t_plot) )] )
                                sigma_l_t_for_plot_legend.append(f"l={l}, j={j}")                            
                                    

                """Save the values to reuse tham in the next cylce on i."""
                rho_t_prec[l][j],drho_t_prec[l][j]=rho_lj_t,drho_lj_t
                
                """We construct the covariance matrix computed at t"""
                diagonal_sigma_l_t.append(1/par.hbar*( 1/rho_lj_t**2-1j*M_t(t)*drho_lj_t/(rho_lj_t) ) )
            
            
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
        comoving_number_area=4*np.pi*np.array(par.n_values)**2#Removed the cut_off
        comving_entanglement_entropy_scaling_t.append([comoving_number_area,entropy_n])
        
    return (comving_entanglement_entropy_scaling_t,max_errors,(sigma_l_t_for_plot,sigma_l_t_for_plot_legend))

# %%CHECK IF THE ANAL SOL IS = TO THE NUMERICAL SOL FOR ERMAKOV-LIKE

if __name__=="__main__":
    
    if par.cosmology=="eds":
        gamma_lj=1
        
        p_lj = 27 * (par.c ** 3) * (cosm.unverse_age ** 2) * (gamma_lj ** 1.5)
        
        def omega_lj2(t):
            """Takes in input the cosmological time x and returns omega_lj2, (\omega_{lj}^2 in the paper),
            which is the frequency appearing in the Ermakov-like equation.
            It needs to be defined globally gamma_l2 and the cosmologcal functions defined in the 
            cosmology module.
            """
            global gamma_lj
            gamma_lj2=gamma_lj**2
            return par.c**2/cosm.scale_factor_t(t)**2*(gamma_lj2+(par.mu*cosm.scale_factor_t(t)*par.cut_off))
        

        def rho_lj_c1c2_(t, C1, C2):
            # Compute θ = ∛(p_lj * t)
            theta = (p_lj * t)**(1/3)
            sqrt_t = t**(1/3)    
            
            # Compute trigonometric functions
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
        
            # Numerator N = [cos(θ) + θ * sin(θ)] * c
            N = (cos_theta + theta * sin_theta) * par.c
        
            # Denominator D = sqrt(C1 * a^3)
            D = np.sqrt(C1 * (t/cosm.unverse_age) ** 2)
        
            # Numerator of F
            num_F = 3 * (p_lj * cos_theta * sqrt_t - p_lj ** (2./3.) * sin_theta)
        
            # Denominator of F
            den_F = p_lj ** 2 * sin_theta * sqrt_t + p_lj ** (5./3.) * cos_theta
        
            # Compute F
            F = num_F / den_F
        
            # Inner term inside the square root
            inner = C2 + C1 * (-F)
        
            # Compute the square root term
            sqrt_term = np.sqrt(par.c ** 4 + inner ** 2)
        
            # Final result
            result = (N / D) * sqrt_term
        
            return result
        
        rho_lj_c1c2=rho_lj_c1c2_
        
        # xtest=np.linspace(par.t_ini, 10,200)
        # ytest=rho_lj_c1c2(xtest,1,1)
        # pl.plot([xtest,ytest])
        
        def rho_ij(t):
            return np.abs(rho_lj_c1c2(t,0.1,1))
        
        
        t_num,rho_lj_t_num,drho_lj_t_num=solve_ermakov_like(t_min=par.t_ini,t_max=par.t_max,
                                                  rho_t_min=rho_ij(par.t_ini),
                                                  drho_t_min=nm.derivative(rho_ij,par.t_ini,1e-6),
                                                  omega_lj2=omega_lj2,
                                                  full_solution=True)
        
        pl.plot([[t_num,rho_lj_t_num],[t_num,rho_ij(t_num)]],
                legend=["Numerical","Analytical"],
                dotted=True,connected_dots=True)
        pl.plot([t_num,rho_ij(t_num)],
                legend=["Analytical"],
                dotted=True,connected_dots=True)

        pl.plot([t_num,rho_lj_t_num],
                legend=["Numerical"],
                dotted=True,connected_dots=True)

        
    elif par.cosmology=="ds":
        gamma_lj=50
                
        def omega_lj2(t):
            """Takes in input the cosmological time x and returns omega_lj2, (\omega_{lj}^2 in the paper),
            which is the frequency appearing in the Ermakov-like equation.
            It needs to be defined globally gamma_l2 and the cosmologcal functions defined in the 
            cosmology module.
            """
            global gamma_lj
            gamma_lj2=gamma_lj**2
            return par.c**2/cosm.scale_factor_t(t)**2*(gamma_lj2+(par.mu*cosm.scale_factor_t(t)*par.cut_off))
        
        
        def rho_lj_c1c2_(t, C1, C2, epsilon=1e-12):
            """
            Calculate rho based on the provided parameters and time.
        
            Parameters:
            - t (float or np.ndarray): Time variable.
            - C1 (float): Constant C1.
            - C2 (float): Constant C2.
            - par: An object or namespace containing constants 'c' and 'hubble_constant'.
            - gamma_lj (float): Gamma_{lj} constant.
            - epsilon (float): Small value to prevent division by zero.
        
            Returns:
            - rho (float or np.ndarray): Calculated rho value.
            """
            # Constants
            c = par.c                   # Speed of light
            Gamma_lj = gamma_lj         # Gamma_{lj}
            H = par.hubble_constant     # Hubble constant
        
            # Check if gamma_lj is effectively zero
            if np.isclose(Gamma_lj, 0.0, atol=epsilon):
                # Handle the gamma_lj = 0 case
                # Compute the exponentials
                exp_3Ht = np.exp(3 * H * t)
                exp_minus6Ht = np.exp(-6 * H * t)
                
                # Compute the denominator inside the square root
                denominator = C1
                
                # Compute the inner terms
                term1 = c**2 * exp_minus6Ht * (exp_3Ht + 1)**2
                term2_inner = C2 - (4 * C1) / (3 * H * exp_3Ht + 3 * H)
                term2 = c**4 + term2_inner**2
                
                # Compute the numerator inside the square root
                numerator = term1 * term2
                
                # Compute the entire expression
                expression = 0.5 * np.sqrt(numerator / denominator)
                
                return expression
        
            else:
                # Handle the gamma_lj ≠ 0 case
                # Exponential terms
                exp_neg_Ht = np.exp(-H * t)
                exp_Ht = np.exp(H * t)
        
                # Argument for sine and cosine functions
                argument = (c * Gamma_lj * exp_neg_Ht) / H
        
                # Sine and cosine terms
                S_term = c * Gamma_lj * np.sin(argument)
                C_term = H * exp_Ht * np.cos(argument)
                sum_SC = S_term + C_term
        
                # First part of the expression
                first_part = (c * exp_neg_Ht * sum_SC) / (np.sqrt(C1) * H)
        
                # Numerator and denominator for the fraction in the second part
                numerator_inner = c * Gamma_lj * np.cos(argument) - H * exp_Ht * np.sin(argument)
                denominator_inner = c * Gamma_lj * np.sin(argument) + H * exp_Ht * np.cos(argument)
        
     
        
                # Fraction within the second part
                fraction = (H**2 * numerator_inner) / (c**3 * Gamma_lj**3 * denominator_inner)
                
                # Total term inside the square root
                total = C2 + C1 * fraction
        
                # Ensure that the argument inside the square root is non-negative
                sqrt_argument = c**4 + total**2
                if np.any(sqrt_argument < 0):
                    raise ValueError("Negative value encountered inside the square root.")
        
                # Second part of the expression
                second_part = np.sqrt(sqrt_argument)
        
                # Final result
                rho = first_part * second_part
        
                return rho
               
      
        
        rho_lj_c1c2=np.vectorize(rho_lj_c1c2_)
        
        # xtest=np.linspace(par.t_ini, 10,200)
        # ytest=rho_lj_c1c2(xtest,1,1)
        # pl.plot([xtest,ytest])
        
        def rho_ij(t):
            return np.abs(rho_lj_c1c2(t,0.1,0))
        
        
        t_num,rho_lj_t_num,drho_lj_t_num=solve_ermakov_like(t_min=par.t_ini,t_max=par.t_max,
                                                  rho_t_min=rho_ij(par.t_ini),
                                                  drho_t_min=nm.derivative(rho_ij,par.t_ini,1e-6),
                                                  omega_lj2=omega_lj2,
                                                  full_solution=True)
        
        pl.plot([[t_num,rho_lj_t_num],[t_num,rho_ij(t_num)]],
                legend=["Numerical","Analytical"],
                dotted=True,connected_dots=True)
        pl.plot([t_num,rho_ij(t_num)],
                legend=["Analytical"],
                dotted=True,connected_dots=True)

        pl.plot([t_num,rho_lj_t_num],
                legend=["Numerical"],
                dotted=True,connected_dots=True)
















