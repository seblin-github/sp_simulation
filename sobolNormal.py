import numpy as np
import math
from scipy.stats import qmc

#Generate 2-dimensional Sobol Sequence
#Box-Muller transform to standard normal
#Reshape to M*N matrix

def generate_sobol_sequences(M, N):
    # Calculate the number of points needed for the Sobol sequence
    num_points = M * N
    num = math.ceil(math.log2(num_points))
    
    # Generate the Sobol sequence
    sobol = qmc.Sobol(d=2, scramble=True)
    seq = sobol.random_base2(num)
    
    return seq

def box_muller_transform(U1, U2):
    # Apply the Box-Muller transform to get standard normal variables
    R = np.sqrt(-2 * np.log(U1))
    theta = 2 * np.pi * U2
    Z1 = R * np.cos(theta)
    Z2 = R * np.sin(theta)
    return Z1, Z2

def generate_standard_normal_matrix(M, N):
    # Generate the Sobol sequences
    sobol_seq = generate_sobol_sequences(M, N)
    
    # Split the Sobol sequence into two arrays for Box-Muller transform
    U1, U2 = sobol_seq[:, 0], sobol_seq[:, 1]
    
    # Apply the Box-Muller transform
    Z1, Z2 = box_muller_transform(U1, U2)
    
    # Combine Z1 and Z2 into a single array
    Z = np.concatenate((Z1, Z2))
    
    # Ensure we have enough elements for the M x N matrix
    if len(Z) < M * N:
        raise ValueError("The generated Sobol sequence does not have enough elements.")
    
    # Reshape the array into an M x N matrix
    standard_normal_matrix = Z[:M * N].reshape(M, N)
    
    return standard_normal_matrix