# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:47:00 2023

@author: Sebastiano Tomasi
"""
import scipy as sp
import numpy as np
from numpy import sqrt, log

import simulation_parameters as par
import cosmology as cosm
import coupling_matrix as cm

import sys
sys.path.append('C:/Users/sebas/Documents/GitHub/custom_libraries')
import plotting_function as pl
import numerical_methods as nm

#%%
"""We first define some functions that we will need in the main cycle."""

def compute_entanglement_entropy(xi):
    """This function takes as input xi, which is a vector containing all xi_i, as
    defined in the paper. It returns the entanglement entropy associated to those."""
    number_of_negative_xi = 0
    s = 0
    for val in xi:
        if par.verbose > 2 and val.imag > 1e-10:
            print(val)
        if val > 0:
            s += -log(1 - val) - val / (1 - val) * log(val)
        elif val < 0:
            if par.verbose >= 4:
                if abs(val) > 1e-10:
                    number_of_negative_xi += 1
                    print(f"Xi is negative! xi={val}<0")
                    
    if par.verbose >= 3 and number_of_negative_xi > 0:
        print(f"Number of negative xi ={number_of_negative_xi}!")
    return s

def M_t(t):
    """This is the effective mass of one oscillator
     when we interpret the discretized Hamiltonian H_{lm} as
     the Hamiltonian for a system of harmonic oscillators. (See the paper.)"""
    return cosm.scale_factor_t(t)**3 / par.c**2

#%%

def ermakov_like_equation_vectorized(t, y, gamma_l2):
    """Vectorized version of the Ermakov-like equation."""
    N = len(gamma_l2)
    rho = y[0:N]
    drho = y[N:2*N]
    
    c = par.c
    a = cosm.scale_factor_t(t)
    H = cosm.hubble_function_t(t)
    muab2 = (par.mu * a * cosm.cut_off_t(t)) ** 2
    
    omega_lj2 = (c / a)**2 * (gamma_l2 + muab2)
    
    dy_dt = np.zeros(2*N)
    dy_dt[0:N] = drho
    dy_dt[N:2*N] = -3 * H * drho - omega_lj2 * rho + c**4 / (a**6 * rho**3)
    return dy_dt

def solve_ermakov_like_vectorized(t_min, t_max, rho_t_min, drho_t_min, gamma_l2, full_solution=True):
    """
    Vectorized version to solve the Ermakov-like equation for multiple gamma_lj2 values at once.
    """
    N = len(gamma_l2)
    y0 = np.concatenate([rho_t_min, drho_t_min])
    # Perform the integration
    rk4_result = sp.integrate.solve_ivp(
        ermakov_like_equation_vectorized, t_span=(t_min, t_max), y0=y0,
        method="RK45", atol=par.ermak_atol, rtol=par.ermak_rtol,
        args=(gamma_l2,)
    )

    if full_solution:
        rho_t = rk4_result.y[0:N, :]
        drho_t = rk4_result.y[N:2*N, :]
        t = rk4_result.t
        return [t, rho_t, drho_t]
    else:
        rho_t_max = rk4_result.y[0:N, -1]
        drho_t_max = rk4_result.y[N:2*N, -1]
        return [rho_t_max, drho_t_max]
    
# %%
def ent_entr_from_slices(A, B, D):
    A_real = np.real(A)
    # Use solve instead of inversion
    A_real_inverse_times_B = sp.linalg.solve(A_real, B)
    
    B_transpose = np.transpose(B)
    B_dagger = np.conjugate(B_transpose)
    
    beta = 0.5 * np.dot(B_dagger, A_real_inverse_times_B)
    gamma = np.real(D - 0.5 * np.dot(B_transpose, A_real_inverse_times_B))
    
    """Define the matrices for the generalized eigenvalue problem"""
    dim = beta.shape[0]
    F = np.block([[-2 * gamma, np.transpose(beta)],
                  [np.eye(dim), np.zeros((dim, dim))]])
    
    G = np.block([[-beta, np.zeros((dim, dim))],
                  [np.zeros((dim, dim)), np.eye(dim)]])
    
    """Solve the generalized eigenvalue problem Fw=\lambda G w"""
    eigenvalues, _ = sp.linalg.eig(F, G)
    
    # We remove infinite eigenvalues
    finite_indices = np.isfinite(eigenvalues)
    num_infinite = np.sum(~finite_indices)
    if par.verbose > 2 and num_infinite > 0:
        print(f"Warning: There are {num_infinite} infinite eigenvalues.")
    finite_eigenvalues = eigenvalues[finite_indices]
    
    # Take the real part of the eigenvalues, since those should already be real. 
    imaginary_parts = np.imag(finite_eigenvalues)
    if par.verbose >= 3 and np.any(imaginary_parts != 0):  # Check if there are non-zero imaginary parts
        max_imaginary_part = np.max(np.abs(imaginary_parts))
        print(f"Warning: The maximum imaginary part being discarded is {max_imaginary_part}.")
    real_parts = np.real(finite_eigenvalues)
    
    # We select only eigenvalues less than 1
    filtered_eigenvalues = real_parts[real_parts < 1]
    
    """The filtered eigenvalues are what are usually called the xi values. We 
    use those to compute the entanglement entropy."""
    return compute_entanglement_entropy(filtered_eigenvalues)

#%%

def collect_plot_data(l, gamma_l2, sigma_l_t_for_plot, sigma_l_t_for_plot_legend):
    """Collect data for plotting sigma_lj_t for specific l and j indices."""
    if l == 0 or l == par.l_max // 2 or l == par.l_max:
        # Indices of j to consider for plotting
        j_indices = [0, par.N // 2, par.N - 1]

        for j in j_indices:
            gamma_lj2_val = gamma_l2[j]  # Extract gamma_lj2_val for specific j

            # Initial conditions for this mode
            a_ini = cosm.scale_factor_t(par.t_ini)
            b_ini = cosm.cut_off_t(par.t_ini)
            muab2_ini = (par.mu * a_ini * b_ini) ** 2
            omega_lj2_ini = (par.c / a_ini) ** 2 * (gamma_lj2_val + muab2_ini)
            M_t_ini = M_t(par.t_ini)
            
            rho_t_min = 1 / np.sqrt(M_t_ini * np.sqrt(omega_lj2_ini))
            drho_t_min = 0.

            # Define gamma_lj2_array and solve the Ermakov-like equation
            gamma_lj2_array = np.array([gamma_lj2_val])
            rho_t_min_array = np.array([rho_t_min])
            drho_t_min_array = np.array([drho_t_min])

            # Solve the Ermakov-like equation vectorized
            t_plot, rho_lj_t_plot_array, drho_lj_t_plot_array = solve_ermakov_like_vectorized(
                t_min=par.t_ini,
                t_max=max(par.times),
                rho_t_min=rho_t_min_array,
                drho_t_min=drho_t_min_array,
                gamma_l2=gamma_lj2_array,
                full_solution=True
            )

            # Extract the solutions
            rho_lj_t_plot = rho_lj_t_plot_array[0]
            drho_lj_t_plot = drho_lj_t_plot_array[0]

            # Compute sigma_lj_t for plotting
            M_t_t_plot = M_t(t_plot)
            sigma_lj_t = 1 / par.hbar * (
                1 / rho_lj_t_plot**2 - 1j * M_t_t_plot * drho_lj_t_plot / rho_lj_t_plot
            )

            # Store results for plotting
            sigma_l_t_for_plot.append([t_plot, sigma_lj_t])
            sigma_l_t_for_plot_legend.append(f"l={l}, j={j}")

def run():
    """Main function to compute entanglement entropy scaling over time and display progress."""
    import time
    total_steps = len(par.times) - 1  # since we start from i=1
    if total_steps <= 0:
        print("No time steps to run.")
        return None
    
    # For progress bar and timing
    iteration_times = []
    
    max_errors = []

    len_n_values = len(par.n_values)
    comoving_entanglement_entropy_scaling_t = []

    rho_t_prec = np.zeros((par.l_max+1, par.N), dtype=float)  # l_max+1 rows and N columns.
    drho_t_prec = np.zeros((par.l_max+1, par.N), dtype=float)
    
    # Precompute gamma_l2 and UT before the time loop.
    gamma_l2_list = []
    UT_list = []
    for l in range(0, par.l_max + 1):
        coupling_matrix_l = cm.generate_ticm(l)
        gamma_l2, UT = sp.linalg.eigh(coupling_matrix_l)
        negative_values = gamma_l2 < 0
        if any(negative_values):
            raise Exception(f"There are some gamma_lj^2 which are negative!:\n {gamma_l2[negative_values]},\n{negative_values}")
        gamma_l2_list.append(gamma_l2)
        UT_list.append(UT)
             
    # Initialize variables for plotting if debug level is high enough
    sigma_l_t_for_plot = []
    sigma_l_t_for_plot_legend = []
    
    # Function to print progress bar and ETA
    def print_progress(iteration, total, elapsed_times):
        bar_length = 18
        progress = iteration / total
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + ' ' * (bar_length - filled_length)
        
        if elapsed_times: 
            avg_time = sum(elapsed_times) / len(elapsed_times)
            remaining = (total - iteration) * avg_time
            eta_str = f"ETA: {int(remaining // 60)}m {int(remaining % 60)}s"
        else:
            eta_str = "ETA: calculating..."
        
        # Print at the same line
        sys.stdout.write(f"\rProgress: [{bar}] {progress*100:.2f}% {eta_str}")
        sys.stdout.flush()

    # Initial print
    print_progress(0, total_steps, [])
    

    for i in range(1, len(par.times)):  # Starts from one because of the definition of times.
        iteration_start = time.time()
        t = par.times[i]
        entropy_nl = np.zeros(len_n_values)
        entropy_n = np.zeros(len_n_values)
        
        
        for l in range(0, par.l_max + 1):  # Cycle on the angular momentum index
            if par.debug_level >= 5:
                if l % 100 == 0:
                    if l == 0:
                        print(f"\nl = {l}, t = {t}")
                    else:
                        print(f"l = {l}")
            
            gamma_l2 = gamma_l2_list[l]
            UT = UT_list[l]
            
            # Initialize initial conditions if i == 1
            if i == 1:
                a_ini = cosm.scale_factor_t(par.t_ini)
                b_ini = cosm.cut_off_t(par.t_ini)
                muab2_ini = (par.mu * a_ini * b_ini) ** 2
                omega_lj2_ini = (par.c / a_ini) ** 2 * (gamma_l2 + muab2_ini)
                M_t_ini = M_t(par.t_ini)
                rho_t_prec[l][:] = 1 / np.sqrt(M_t_ini * np.sqrt(omega_lj2_ini))
                drho_t_prec[l][:] = 0.
                
                # Collect data for plotting if debug level is high enough
                if par.debug_level >= 3:
                    collect_plot_data(l, gamma_l2, sigma_l_t_for_plot, sigma_l_t_for_plot_legend)
            
            # Solve the Ermakov-like equation vectorized
            rho_l_t, drho_l_t = solve_ermakov_like_vectorized(
                t_min=par.times[i-1], t_max=t,
                rho_t_min=rho_t_prec[l][:], drho_t_min=drho_t_prec[l][:],
                gamma_l2=gamma_l2,
                full_solution=False)

            # Correct sign changes
            sign_change = np.sign(rho_l_t) != np.sign(rho_t_prec[l][:])
            rho_l_t[sign_change] *= -1
            drho_l_t[sign_change] *= -1
            if par.verbose >= 0 and np.any(sign_change):
                indices = np.where(sign_change)[0]
                print(f"Sign change in rho detected and corrected at l={l}, t={t} for indices {indices}")

            # Update rho_t_prec and drho_t_prec
            rho_t_prec[l][:] = rho_l_t
            drho_t_prec[l][:] = drho_l_t

            # Compute diagonal_sigma_l_t
            M_t_t = M_t(t)
            diagonal_sigma_l_t = 1 / par.hbar * (1 / rho_l_t**2 - 1j * M_t_t * drho_l_t / rho_l_t)

            # Construct sigma_l_t and omega_l_t
            sigma_l_t = np.diag(diagonal_sigma_l_t)
            omega_l_t = UT @ sigma_l_t @ UT.T  # Note that U is real, so U^T = U^{-1}

            for k, n in enumerate(par.n_values):
                if n == par.n_min or n == par.n_max:
                    """If we consider the complete system or no system
                    the entanglement entropy must be zero."""
                    pass
                else:
                    """Compute the entanglement entropy from the sliced omega_l_t"""
                    A, B, B_T, D = cm.slice_matrix(omega_l_t, n)  # In order to trace out n oscillators.
                    entropy_nl[k] = ent_entr_from_slices(A, B, D)
            
            if l == par.l_max:
                nonzero_values = entropy_nl != 0
                nonzero_entropy_nl = entropy_nl[nonzero_values]
                nonzero_entropy_n = entropy_n[nonzero_values]
                # max_error defined as |S(l_max)-S(l_max-1)|/(S(l_max)+S(l_max-1))/2
                if len(nonzero_entropy_nl) > 0:
                    max_error = np.max(1 / (0.5 + nonzero_entropy_n / ((2 * l + 1) * nonzero_entropy_nl)))
                    max_errors.append(max_error)
                    if par.debug_level >= 5:
                        print("Maximum relative deviation = {:.4e}".format(max_error))
                    print("")
            
            entropy_n = entropy_n + (2 * l + 1) * entropy_nl

        comoving_number_area = 4 * np.pi * np.array(par.n_values)**2  # Removed the cut_off
        comoving_entanglement_entropy_scaling_t.append([comoving_number_area, entropy_n])

        # Timing and progress bar update
        iteration_end = time.time()
        iteration_time = iteration_end - iteration_start
        iteration_times.append(iteration_time)

        # Update progress bar and ETA after each iteration
        print_progress(i, total_steps, iteration_times)

    # Print newline at the end of progress
    print("\nComputation completed!")
    return (comoving_entanglement_entropy_scaling_t, max_errors, (sigma_l_t_for_plot, sigma_l_t_for_plot_legend))


# %%CHECK IF THE ANALYTICAL SOLUTION IS EQUAL TO THE NUMERICAL SOLUTION FOR ERMAKOV-LIKE

if __name__ == "__main__":
    
    if par.cosmology == "eds":
        gamma_lj = 1
        gamma_l2 = np.array([gamma_lj ** 2])  # Make it an array for the vectorized solver
        
        p_lj = 27 * (par.c ** 3) * (cosm.universe_age ** 2) * (gamma_lj ** 1.5)
        
        def rho_lj_c1c2_(t, C1, C2):
            # Compute θ = ∛(p_lj * t)
            theta = (p_lj * t) ** (1 / 3)
            sqrt_t = t ** (1 / 3)
            
            # Compute trigonometric functions
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
        
            # Numerator N = [cos(θ) + θ * sin(θ)] * c
            N = (cos_theta + theta * sin_theta) * par.c
        
            # Denominator D = sqrt(C1 * a^3)
            D = np.sqrt(C1 * (t / cosm.universe_age) ** 2)
        
            # Numerator of F
            num_F = 3 * (p_lj * cos_theta * sqrt_t - p_lj ** (2. / 3.) * sin_theta)
        
            # Denominator of F
            den_F = p_lj ** 2 * sin_theta * sqrt_t + p_lj ** (5. / 3.) * cos_theta
        
            # Compute F
            F = num_F / den_F
        
            # Inner term inside the square root
            inner = C2 + C1 * (-F)
        
            # Compute the square root term
            sqrt_term = np.sqrt(par.c ** 4 + inner ** 2)
        
            # Final result
            result = (N / D) * sqrt_term
        
            return result
        
        rho_lj_c1c2 = rho_lj_c1c2_
        
        def rho_ij(t):
            return np.abs(rho_lj_c1c2(t, 1, 1))
        
        # Initial conditions
        rho_t_min = np.array([rho_ij(par.t_ini)])
        drho_t_min = np.array([nm.derivative(rho_ij, par.t_ini, 1e-6)])
        
        # Solve the Ermakov-like equation using the vectorized solver
        t_num, rho_lj_t_array, drho_lj_t_array = solve_ermakov_like_vectorized(
            t_min=par.t_ini,
            t_max=par.t_max,
            rho_t_min=rho_t_min,
            drho_t_min=drho_t_min,
            gamma_l2=gamma_l2,
            full_solution=True
        )
        
        # Extract the numerical solution
        rho_lj_t_num = rho_lj_t_array[0]
        drho_lj_t_num = drho_lj_t_array[0]
        
        # Analytical solution
        rho_lj_t_analytical = rho_ij(t_num)
        
        # Plot the results
        pl.plot([[t_num, rho_lj_t_analytical]],
                legend=["Numerical", "Analytical"],
                # xlim=(0.1,0.11),
                dotted=True, connected_dots=True)
    
    elif par.cosmology == "ds":
        gamma_lj = 0.1
        gamma_l2 = np.array([gamma_lj ** 2])  # Make it an array for the vectorized solver
        
        def rho_lj_c1c2_(t, C1, C2, epsilon=1e-12):
            """
            Calculate rho based on the provided parameters and time.
            
            Parameters:
            - t (float or np.ndarray): Time variable.
            - C1 (float): Constant C1.
            - C2 (float): Constant C2.
            - epsilon (float): Small value to prevent division by zero.
            
            Returns:
            - rho (float or np.ndarray): Calculated rho value.
            """
            # Constants
            c = par.c                   # Speed of light
            Gamma_lj = gamma_lj         # Gamma_{lj}
            H = par.ds_hubble_constant     # Hubble constant
            
            # Check if gamma_lj is effectively zero
            if np.isclose(Gamma_lj, 0.0, atol=epsilon):
                # Handle the gamma_lj = 0 case
                # Compute the exponentials
                exp_3Ht = np.exp(3 * H * t)
                exp_minus6Ht = np.exp(-6 * H * t)
                
                # Compute the denominator inside the square root
                denominator = C1
                
                # Compute the inner terms
                term1 = c ** 2 * exp_minus6Ht * (exp_3Ht + 1) ** 2
                term2_inner = C2 - (4 * C1) / (3 * H * exp_3Ht + 3 * H)
                term2 = c ** 4 + term2_inner ** 2
                
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
                fraction = (H ** 2 * numerator_inner) / (c ** 3 * Gamma_lj ** 3 * denominator_inner)
                
                # Total term inside the square root
                total = C2 + C1 * fraction
        
                # Ensure that the argument inside the square root is non-negative
                sqrt_argument = c ** 4 + total ** 2
                if np.any(sqrt_argument < 0):
                    raise ValueError("Negative value encountered inside the square root.")
        
                # Second part of the expression
                second_part = np.sqrt(sqrt_argument)
        
                # Final result
                rho = first_part * second_part
        
                return rho
        
        rho_lj_c1c2 = np.vectorize(rho_lj_c1c2_)
        
        def rho_ij(t):
            return np.abs(rho_lj_c1c2(t, 0.1, 0))
        
        # Initial conditions
        rho_t_min = np.array([rho_ij(par.t_ini)])
        drho_t_min = np.array([nm.derivative(rho_ij, par.t_ini, 1e-6)])
        
        # Solve the Ermakov-like equation using the vectorized solver
        t_num, rho_lj_t_array, drho_lj_t_array = solve_ermakov_like_vectorized(
            t_min=par.t_ini,
            t_max=par.t_max,
            rho_t_min=rho_t_min,
            drho_t_min=drho_t_min,
            gamma_l2=gamma_l2,
            full_solution=True
        )
        
        # Extract the numerical solution
        rho_lj_t_num = rho_lj_t_array[0]
        drho_lj_t_num = drho_lj_t_array[0]
        
        # Analytical solution
        rho_lj_t_analytical = rho_ij(t_num)
        
        # Plot the results
        pl.plot([[t_num, np.abs(rho_lj_t_num)], [t_num, rho_lj_t_analytical]],
                legend=["Numerical", "Analytical"],
                dotted=True, connected_dots=True)
