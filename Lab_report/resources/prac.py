import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

array2d = np.loadtxt("Workshop_data_Week17.csv", delimiter=',')
new_array2d = array2d[484:]

Time_3 = array2d[:,0]
Voltage_3 = new_array2d[:,1:]
print(Voltage_3)
plt.hist(Voltage_3)
plt.xlabel('Voltages /$V$')
plt.ylabel('Frequency')
plt.title("Distribution of random voltages")
plt.show()

