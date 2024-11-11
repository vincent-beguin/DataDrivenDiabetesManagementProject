import numpy as np
import sklearn as skl

#Import data from csv file
data = np.genfromtxt("diabetes.csv",delimiter=',')

x_data = data[1:, :7]
y_data = data[1:, 8]

print(x_data)
print(y_data)

