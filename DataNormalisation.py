import numpy as np

norm_factors = np.array([1/20, 1/200, 1/150, 1/100, 1/1000, 1/100, 1/3, 1/100])

def data_normalisation(data):
    norm_matrix = np.diagonal(norm_factors)
    norm_data = np.matmul(norm_matrix, data)
    return norm_data
