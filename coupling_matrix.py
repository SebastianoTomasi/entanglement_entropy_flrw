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
is_flat_flrw = True if par.cosmology in ["eds","ds","lcdm","rad","flat"] else False
    
if par.use_midpoint_scheme and is_flat_flrw:
    def diagonal_elements(l):
        res=[]
        start,stop=par.n_min+1,par.n_max+1
        kb2=0
        for j in range(start,stop):
            
            """Can impose different boundary conditions using the conditionals 
            at j==1 and j==par.n_max"""
            if j==1:
                res.append(9/4*np.sqrt((1-9/4*kb2)*(1-kb2))+l*(l+1))
            elif j==par.n_max:
                res.append((j - 0.5)**2/ j**2 * np.sqrt(1 - kb2 * (j - 0.5)**2) * np.sqrt(1 - kb2 * j**2) )
            else:
                res.append((((1 - kb2 * j**2)**0.5) / (j**2)) * (((j + 0.5)**2) * ((1 - kb2 * (j + 0.5)**2)**0.5) + ((j - 0.5)**2) * ((1 - kb2 * (j - 0.5)**2)**0.5)) + (l * (l + 1) / (j**2)))
        return res
    
    def top_diagonal_elements():
        kb2=0
        j=np.arange(par.n_min+2,par.n_max+1)
        return -((j + 0.5)**2 / (j * (j + 1))) * (((1 -kb2 * (j + 0.5)**2) * ((1 -kb2 * j**2) * (1 -kb2 * (j + 1)**2))**0.5)**0.5)

    
elif not par.use_midpoint_scheme and is_flat_flrw:
    def diagonal_elements(l):
        res=[]
        start,stop=par.n_min+1,par.n_max+1
        for j in range(start,stop):
            """Can impose different boundary conditions using the conditionals 
            at j==1 and j==par.n_max"""
            if j==1:
                res.append( 1 + l*(l+1) )
            elif j==par.n_max:
                res.append((j-1)**2/j**2)
            else:
                res.append(2 + (1-2*j)/j**2 + l*(l+1)/j**2 )
        return res
    
    def top_diagonal_elements():
        j=np.arange(par.n_min+2,par.n_max+1)
        return -1+1/(1+j)


# %% NONZERO SPATIAL CURVATURE without the midpoint stuff
elif not par.use_midpoint_scheme and par.cosmology=="snyder":
    def diagonal_elements(l):
        res=[]
        start,stop=par.n_min+1,par.n_max+1
        kb2=par.k*par.cut_off**2
       
        for j in range(start,stop):
            
            """Can impose different boundary conditions using the conditionals 
            at j==1 and j==par.n_max"""
            if j==1:
                res.append( 1 + l*(l+1)-kb2)
            elif j==par.n_max:
                res.append(((j - 1)**2 / j**2) * np.sqrt((1 - kb2 * j**2) * (1 - kb2 * (j - 1)**2)))
            else:
                res.append(1 + (l * (l + 1)) / j**2 - kb2* j**2 + ((j - 1)**2 / j**2) * np.sqrt((1 - kb2 * j**2) * (1 - kb2 * (j - 1)**2))
                       )
        return res
    
    def top_diagonal_elements():
        kb2=par.k*par.cut_off**2
        j=np.arange(par.n_min+2,par.n_max+1)
        return -(j / (j + 1)) * (1 - kb2 * j**2)**(3/4) * (1 - kb2 * (j + 1)**2)**(1/4)
    
# %% NONZERO SPATIAL CURVATURE with midpoint approx

elif par.use_midpoint_scheme and par.cosmology=="snyder":
    def diagonal_elements(l):
        res=[]
        start,stop=par.n_min+1,par.n_max+1
        kb2=par.k*par.cut_off**2
        for j in range(start,stop):
            
            """Can impose different boundary conditions using the conditionals 
            at j==1 and j==par.n_max"""
            if j==1:
                res.append(9/4*np.sqrt((1-9/4*kb2)*(1-kb2))+l*(l+1))
            elif j==par.n_max:
                res.append((j - 0.5)**2/ j**2 * np.sqrt(1 - kb2 * (j - 0.5)**2) * np.sqrt(1 - kb2 * j**2) )
            else:
                res.append((((1 - kb2 * j**2)**0.5) / (j**2)) * (((j + 0.5)**2) * ((1 - kb2 * (j + 0.5)**2)**0.5) + ((j - 0.5)**2) * ((1 - kb2 * (j - 0.5)**2)**0.5)) + (l * (l + 1) / (j**2)))
        return res
    
    def top_diagonal_elements():
        kb2=par.k*par.cut_off**2
        j=np.arange(par.n_min+2,par.n_max+1)
        return -((j + 0.5)**2 / (j * (j + 1))) * (((1 -kb2 * (j + 0.5)**2) * ((1 -kb2 * j**2) * (1 -kb2 * (j + 1)**2))**0.5)**0.5)

else:
    raise Exception(f"No coupling matrix defined for cosmology={par.cosmology}")
    
def down_diagonal_elements():
    return top_diagonal_elements()
# %%


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