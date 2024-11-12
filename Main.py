import numpy as np
import sklearn as skl
from DataNormalisation import data_normalisation

#Import data from csv file
data = np.genfromtxt("diabetes.csv",delimiter=',')

x_data = data[1:, :7]
y_data = data[1:,  8]

#Data normalisation


#Split the data into a training set and a validation set. This section might get moved to TrainTestSplit.py if the code gets longer

x_train, x_valid, y_train, y_valid = skl.train_test_split(x_data, y_data, test_size=0.2, random_state=1) #80-20 split. random_state is set to 1 to keep repeatability between tests (for now).
