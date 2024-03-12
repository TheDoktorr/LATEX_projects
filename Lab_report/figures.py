import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#   array_binned = np.array([[0,1,2,3,4,5,6,7,8],[1,14,25,18,20,11,6,4,1]])
#   np.savetxt("binned_data.csv", array_binned, delimiter=",")

list_new = np.loadtxt("bg_raw.csv")
print(list_new)


