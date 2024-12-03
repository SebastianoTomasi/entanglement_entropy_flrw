# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:20:11 2024

@author: Sebastiano Tomasi
"""
import simulation_parameters as par
import numpy as np


"""In this module we write the functions used to define the time independent coupling matrix."""
#%%
"""Here we define the entries of the time independent part of the coupling matrix"""
# %% ZERO SPATIAL CURVATURE 
# def diagonal_elements(l):
#     res=[]
#     start,stop=par.n_min+1,par.n_max+1
#     for i in range(start,stop):
#         """Can impose different boundary conditions using the conditionals 
#         at i==1 and i==par.n_max"""
#         if i==1:
#             res.append( 9/4 + l*(l+1) )
#         # if i==par.n_max:
#         #     # res.append((par.n_max-0.5)**2/par.n_max**2)
#         #     print((par.n_max-0.5)**2/par.n_max**2-(2 + (l*(l+1)+0.5)/i**2 +muab_t2))
#         #     res.append(2 + (l*(l+1)+0.5)/i**2 +muab_t2)
#         else:
#             res.append(2 + (l*(l+1)+0.5)/i**2 )
#     return res

# def top_diagonal_elements():
#     i=np.arange(par.n_min+2,par.n_max+1)
#     return -1-1/(4*i*(i+1))

# def down_diagonal_elements():
#     return top_diagonal_elements()
# %% NONZERO SPATIAL CURVATURE AND NO +1/2 

# def diagonal_elements(l):
#     res=[]
#     start,stop=par.n_min+1,par.n_max+1
#     kb2=par.k*par.cut_off**2
#     for i in range(start,stop):
#         """Can impose different boundary conditions using the conditionals 
#         at i==1 and i==par.n_max"""
#         if i==1:
#             res.append( 1/(1-kb2) + l*(l+1) )
#         # if i==par.n_max:
#         #     res.append((i-1)**2/(i**2(1-kb2*(i-1)**2)))
#         else:
#             res.append( 1/(1-kb2*i**2) + (i-1)**2/(1-kb2*(i-1)**2)/i**2 + l*(l+1)/i**2 )
#     return res

# def top_diagonal_elements():
#     kb2=par.k*par.cut_off**2
#     i=np.arange(par.n_min+2,par.n_max+1)
#     return i/(i+1)/(1-kb2*i**2)

# def down_diagonal_elements():
#     return top_diagonal_elements()

# %% NONZERO SPATIAL CURVATURE with the +1/2 

def diagonal_elements(l):
    res=[]
    start,stop=par.n_min+1,par.n_max+1
    kb2=par.k*par.cut_off**2
    for i in range(start,stop):
        """Can impose different boundary conditions using the conditionals 
        at i==1 and i==par.n_max"""
        if i==1:
            res.append( (9 / 4) / (1 - (9 / 4) * kb2) + l * (l + 1) )
        # elif i==par.n_max:
        #     res.append((i-0.5)**2/((1-kb2*(i-1/2)**2)*i**2) )
        else:
            res.append( (1 / i**2) * (
                ((i + 0.5)**2) / (1 - kb2 * (i + 0.5)**2) +
                ((i - 0.5)**2) / (1 - kb2 * (i - 0.5)**2) +
                l * (l + 1)
            ) )
    return res

def top_diagonal_elements():
    kb2=par.k*par.cut_off**2
    i=np.arange(par.n_min+2,par.n_max+1)
    return ((i + 0.5)**2) / ((1 - kb2 * (i + 0.5)**2) * i * (i + 1))

def down_diagonal_elements():
    return top_diagonal_elements()


def generate_ticm(l):#generate time independent coupling matrix
    """Generates the time independent coupling matrix, corresponding
    to a fixed value of the spherical harmonic index l. """
    diagonal = diagonal_elements(l)
    top_diagonal =top_diagonal_elements()#The off diagonals are shorter then the diagonal.
    down_diagonal=down_diagonal_elements()

    """Create the coupling matrix"""
    coupling_matrix = np.diag(diagonal) + np.diag(top_diagonal, 1) + np.diag(down_diagonal, -1)
    return coupling_matrix

def slice_matrix(matrix, n):
    """
    Slice a square matrix into four blocks.
    
    The function divides the input matrix into four blocks:
    - diag_up: a square sub-matrix of dimension nxn in the upper left corner
    - off_diag_up: a rectangular sub-matrix in the upper right corner
    - diag_down: a square sub-matrix in the lower right corner
    - off_diag_down: a rectangular sub-matrix in the lower left corner
    
    Parameters:
    matrix (numpy.ndarray): The input square matrix.
    n (int): The dimension for the 'diag_up' block. Must be less than or equal to the dimension of 'matrix'.
    
    Returns:
    list: A 2x2 matrix whose elements are the sub-matrix blocks.
    """
    
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The input matrix must be square.")
    
    # Check if n is less than or equal to the dimension of the matrix
    if n > matrix.shape[0]:
        raise ValueError(f"n ({n}) is greater than the dimension of the matrix ({matrix.shape[0]}).")
    
    # Slice the matrix into four blocks
    diag_up = matrix[:n, :n]
    off_diag_up = matrix[:n, n:]
    diag_down = matrix[n:, n:]
    off_diag_down = matrix[n:, :n]
    
    return [diag_up, off_diag_up,off_diag_down,diag_down]