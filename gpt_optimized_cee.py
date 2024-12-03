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

"""
Optimizations Applied:
    - Vectorized ODE solving for the Ermakov-like equation.
    - Precomputed eigenvalues and eigenvectors outside the time loop.
    - Optimized entropy calculation using efficient linear algebra operations.
    - Removed unnecessary loops and redundant computations.
    - Adjusted ODE solver to handle non-vectorized function.
"""

#%%

def compute_entanglement_entropy(xi):
    """Compute the entanglement entropy from xi values."""
    s = 0
    for val in xi:
        if val > 0:
            s += -log(1 - val) - val / (1 - val) * log(val)
        elif val < 0 and par.verbose >= 4 and abs(val) > 1e-10:
            print(f"Xi is negative! xi={val}<0")
    return s

def M_t(t):
    """Effective mass M(t) of one oscillator."""
    return cosm.scale_factor_t(t)**3 / par.c**2

#%%

def ermakov_like_equation(t, y, gamma_lj2):
    """
    Ermakov-like equation for all gamma_lj2.

    Parameters:
    - t: Time variable.
    - y: Array containing rho and drho for all gamma_lj2.
    - gamma_lj2: Array of eigenvalues gamma_lj2.

    Returns:
    - dydt: Derivative of y with respect to time.
    """
    N = len(gamma_lj2)
    rho = y[:N]
    drho = y[N:]
    omega2 = omega_lj2_vec(t, gamma_lj2)
    a_t = cosm.scale_factor_t(t)
    h_t = cosm.hubble_function_t(t)
    ddrho = -3 * h_t * drho - omega2 * rho + par.c**4 / (a_t**6 * rho**3)
    dydt = np.concatenate([drho, ddrho])
    return dydt

def omega_lj2_vec(t, gamma_lj2):
    """
    Vectorized omega_lj2 function.

    Parameters:
    - t: Time variable.
    - gamma_lj2: Array of eigenvalues gamma_lj2.

    Returns:
    - omega_lj2: Array of omega_lj2 values.
    """
    a = cosm.scale_factor_t(t)
    b = cosm.cut_off_t(t)
    mu = par.mu
    c = par.c
    muab2 = (mu * a * b)**2
    return (c / a)**2 * (gamma_lj2 + muab2)

def solve_ermakov_like_vectorized(t_min, t_max, rho_t_min, drho_t_min, gamma_lj2):
    """
    Solver for the Ermakov-like equation.

    Parameters:
    - t_min: Initial time.
    - t_max: Final time.
    - rho_t_min: Initial rho values.
    - drho_t_min: Initial drho values.
    - gamma_lj2: Array of eigenvalues gamma_lj2.

    Returns:
    - rho_t: rho values at t_max.
    - drho_t: drho values at t_max.
    """
    y0 = np.concatenate([rho_t_min, drho_t_min])  # Initial conditions

    sol = sp.integrate.solve_ivp(
        lambda t, y: ermakov_like_equation(t, y, gamma_lj2),
        [t_min, t_max],
        y0,
        method='RK45',
        vectorized=False,  # Set vectorized=False to handle y as 1D array
        rtol=par.ermak_rtol,
        atol=par.ermak_atol
    )

    rho_t = sol.y[:par.N, -1]
    drho_t = sol.y[par.N:, -1]
    return rho_t, drho_t

def compute_entropy(A, B, D):
    """
    Compute the entanglement entropy for given matrices.

    Parameters:
    - A: Matrix A from omega_l_t.
    - B: Matrix B from omega_l_t.
    - D: Matrix D from omega_l_t.

    Returns:
    - entropy: Calculated entropy.
    """
    A_real = np.real(A)
    # Use linear solver instead of explicit inversion
    A_real_inv_B = sp.linalg.solve(A_real, B, assume_a='sym')
    B_dagger = B.conj().T
    beta = 0.5 * (B_dagger @ A_real_inv_B)
    gamma = np.real(D - 0.5 * (B.T @ A_real_inv_B))

    # Construct F and G efficiently
    dim = beta.shape[0]
    F = np.block([
        [-2 * gamma, beta.T],
        [np.eye(dim), np.zeros((dim, dim))]
    ])
    G = np.block([
        [-beta, np.zeros((dim, dim))],
        [np.zeros((dim, dim)), np.eye(dim)]
    ])

    # Solve the generalized eigenvalue problem
    eigenvalues = sp.linalg.eigvals(F, G, overwrite_a=True, check_finite=True)

    # Filter eigenvalues
    finite_eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    real_parts = np.real(finite_eigenvalues)
    xi_values = real_parts[real_parts < 1]

    # Compute entropy
    entropy = compute_entanglement_entropy(xi_values)
    return entropy

#%%

def run():
    """
    Main function to compute entanglement entropy scaling over time.

    Steps:
    1. Precompute eigenvalues and eigenvectors.
    2. Initialize variables.
    3. Loop over time steps.
    4. ODE solving for rho and drho.
    5. Compute ground state matrices and entanglement entropy.
    6. Collect and store results.
    """
    len_n_values = len(par.n_values)
    comoving_entanglement_entropy_scaling_t = []

    # Precompute gamma_l2 and U for all l
    gamma_l2_list = []
    U_list = []
    for l in range(par.l_max + 1):
        coupling_matrix_l = cm.generate_ticm(l)
        gamma_l2, UT = sp.linalg.eigh(coupling_matrix_l)
        gamma_l2_list.append(gamma_l2)
        U_list.append(UT.T)

    # Initialize rho and drho
    rho_t_prec = np.zeros((par.l_max + 1, par.N))
    drho_t_prec = np.zeros((par.l_max + 1, par.N))

    for i in range(1, len(par.times)):
        t = par.times[i]
        t_prev = par.times[i - 1]
        entropy_n = np.zeros(len_n_values)

        for l in range(par.l_max + 1):
            if par.debug_level >= 1 and l % 100 == 0:
                if l == 0:
                    print(f"l = {l}, t = {t}")
                else:
                    print(f"l = {l}")

            gamma_l2 = gamma_l2_list[l]
            U = U_list[l]

            # Initial conditions
            if i == 1:
                omega_lj2_ini = omega_lj2_vec(par.t_ini, gamma_l2)
                rho_t_prec[l] = 1 / np.sqrt(M_t(par.t_ini) * np.sqrt(omega_lj2_ini))
                drho_t_prec[l] = np.zeros(par.N)

            # ODE solving
            rho_l_t, drho_l_t = solve_ermakov_like_vectorized(
                t_min=t_prev,
                t_max=t,
                rho_t_min=rho_t_prec[l],
                drho_t_min=drho_t_prec[l],
                gamma_lj2=gamma_l2
            )

            # Correct sign changes if necessary
            sign_change = np.sign(rho_l_t) != np.sign(rho_t_prec[l])
            if np.any(sign_change):
                rho_l_t[sign_change] *= -1
                drho_l_t[sign_change] *= -1
                if par.verbose >= 0:
                    print(f"Sign change in rho detected and corrected for l={l} at t={t}")

            # Update previous rho and drho
            rho_t_prec[l] = rho_l_t
            drho_t_prec[l] = drho_l_t

            # Construct ground state matrix
            diagonal_sigma_l_t = 1 / par.hbar * (1 / rho_l_t**2 - 1j * M_t(t) * drho_l_t / rho_l_t)
            sigma_l_t = np.diag(diagonal_sigma_l_t)
            omega_l_t = U.T @ sigma_l_t @ U  # Use matrix multiplication operator

            # Compute entanglement entropy
            entropy_nl = np.zeros(len_n_values)
            for k, n in enumerate(par.n_values):
                if n == par.n_min or n == par.n_max:
                    continue  # Entropy is zero for full or empty system
                else:
                    A, B, B_T, D = cm.slice_matrix(omega_l_t, n)
                    entropy_nl[k] = compute_entropy(A, B, D)

            # Accumulate entropy
            entropy_n += (2 * l + 1) * entropy_nl

        # Store results
        comoving_number_area = 4 * np.pi * np.array(par.n_values)**2
        comoving_entanglement_entropy_scaling_t.append([comoving_number_area, entropy_n])

    return (comoving_entanglement_entropy_scaling_t,[],([],[]))

# %%CHECK IF THE ANAL SOL IS = TO THE NUMERICAL SOL FOR ERMAKOV-LIKE

if __name__ == "__main__":
    # Test code to verify the Ermakov-like equation solver
    if par.cosmology == "eds":
        gamma_lj = 1

        def omega_lj2(t):
            a = cosm.scale_factor_t(t)
            b = cosm.cut_off_t(t)
            mu = par.mu
            c = par.c
            muab2 = (mu * a * b)**2
            return (c / a)**2 * (gamma_lj**2 + muab2)

        def rho_ij(t):
            # Placeholder analytical solution
            return np.abs(1 / np.sqrt(M_t(t) * np.sqrt(omega_lj2(t))))

        t_num = np.linspace(par.t_ini, par.t_max, 100)
        rho_lj_t_num = np.zeros((par.N, len(t_num)))
        drho_lj_t_num = np.zeros((par.N, len(t_num)))
        for idx, t_val in enumerate(t_num):
            rho_t, drho_t = solve_ermakov_like_vectorized(
                t_min=par.t_ini,
                t_max=t_val,
                rho_t_min=rho_ij(par.t_ini),
                drho_t_min=nm.derivative(rho_ij, par.t_ini, 1e-6),
                gamma_lj2=np.array([gamma_lj**2])
            )
            rho_lj_t_num[:, idx] = rho_t
            drho_lj_t_num[:, idx] = drho_t

        pl.plot([[t_num, rho_lj_t_num[0]], [t_num, rho_ij(t_num)]],
                legend=["Numerical", "Analytical"],
                dotted=True, connected_dots=True)
    elif par.cosmology == "ds":
        gamma_lj = 50

        def omega_lj2(t):
            a = cosm.scale_factor_t(t)
            b = cosm.cut_off_t(t)
            mu = par.mu
            c = par.c
            muab2 = (mu * a * b)**2
            return (c / a)**2 * (gamma_lj**2 + muab2)

        def rho_ij(t):
            # Placeholder analytical solution
            return np.abs(1 / np.sqrt(M_t(t) * np.sqrt(omega_lj2(t))))

        t_num = np.linspace(par.t_ini, par.t_max, 100)
        rho_lj_t_num = np.zeros((par.N, len(t_num)))
        drho_lj_t_num = np.zeros((par.N, len(t_num)))
        for idx, t_val in enumerate(t_num):
            rho_t, drho_t = solve_ermakov_like_vectorized(
                t_min=par.t_ini,
                t_max=t_val,
                rho_t_min=rho_ij(par.t_ini),
                drho_t_min=nm.derivative(rho_ij, par.t_ini, 1e-6),
                gamma_lj2=np.array([gamma_lj**2])
            )
            rho_lj_t_num[:, idx] = rho_t
            drho_lj_t_num[:, idx] = drho_t

        pl.plot([[t_num, rho_lj_t_num[0]], [t_num, rho_ij(t_num)]],
                legend=["Numerical", "Analytical"],
                dotted=True, connected_dots=True)
