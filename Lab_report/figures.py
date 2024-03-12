import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import poisson

array_new = np.loadtxt('bg_raw.prn')
print(array_new)

plt.figure(figsize=(10,10))
plt.hist(array_new, range=(0,8), bins=9)
plt.xlabel("Count")
plt.ylabel("frequency")
plt.title("Counts in a 3s interval")
# plt.savefig('plain_hist.svg') ->> run every so often
# plt.show()

mean_raw = np.mean(array_new)
print(mean_raw)

def fit_func(k, lamb):
    return poisson.pmf(k, lamb)
parameters, cov_matrix = curve_fit(fit_func)