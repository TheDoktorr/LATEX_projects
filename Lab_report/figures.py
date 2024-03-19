import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.special import factorial
import matplotlib.ticker as ticker


array_new = np.loadtxt('bg_raw.prn')
print(array_new)

def fit_func(k, lamb):
    return poisson.pmf(k, lamb)

bins = np.arange(10) - 0.5
print(bins)

plt.figure(figsize=(10,10))
entries, bin_edges, patches = plt.hist(array_new, range=(0,8), bins=bins,density=True, label="Counts")
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])


plt.xlabel("Count")
plt.ylabel("frequency")
plt.title("Counts in a 3s interval")

mean_raw = np.mean(array_new)
print(mean_raw)

parameters, cov_matrix = curve_fit(fit_func, bin_centers, entries)

x = np.arange(0, 10)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.plot(x,
         fit_func(x, *parameters),
         marker='o',
         label='fit',
)

plt.legend()
plt.tight_layout()
plt.savefig('histogram_new.svg')
plt.show()


print(cov_matrix)

# https://stackoverflow.com/questions/25828184/fitting-to-poisson-histogram
# https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
# https://matplotlib.org/stable/users/explain/axes/axes_ticks.html
# https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-the-x-or-y-axis