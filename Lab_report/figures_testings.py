import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.special import factorial
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline, BSpline
from scipy.stats import rv_discrete
from scipy import stats
from scipy.stats import fit

class poisson_gen(rv_discrete):
    def _pmf(self, k, mu):

        return np.exp(-mu) * mu**k / factorial(k)
poisson = poisson_gen(name="poisson")


array_new = np.loadtxt('bg_raw.prn')
print(array_new)

def fit_func(k, lamb):
    return poisson.pmf(k, lamb)


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.set_xlim(-0.5,9.5)
ax1.set_ylim(0,27)
ax2.set_ylim(0, 0.27)

x = np.arange(0, 10)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))

bins = np.arange(10) - 0.5
color = 'tab:blue'
ax1.set_xlabel('Count')
ax1.set_ylabel('frequency', color=color)

entries, bin_edges, patches =ax1.hist(array_new, histtype='step', fill='', range=(0,8), bins=bins, density=False, color=color, label="Counts")

bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
ax1.tick_params(axis='y', labelcolor = color)

height, _ = np.histogram(array_new, range=(0,8), bins=bins, density=False)
print(x.shape)
print(height.shape)
print(height)

ax1.errorbar(bin_centers,height,xerr=0, yerr=np.sqrt(height), fmt='', linestyle="None", capsize=2)
# print(bins)
parameters, cov_matrix = curve_fit(fit_func, bin_centers, entries)
print(parameters)


# data = int(array_new)

dist = stats.poisson
bounds = [(3, 4)]
res = stats.fit(dist, array_new, bounds)

xnew = np.linspace(x.min(), x.max(), 300)

print(res.params)
mu = res.params[0]
print(mu)
spl = make_interp_spline(x, fit_func(x, mu), k=3)
beans = spl(xnew)


# res.plot()
color ='tab:red'
ax2.set_ylabel('probability', color=color)
ax2.plot(x, fit_func(x, mu), marker='o', color = color, ls = '', ms = 4.0, )
ax2.plot(xnew, beans, marker=',', label='fit', color = color)
ax2.tick_params(axis='y', labelcolor = color)


# mean_raw = np.mean(array_new)
# print(mean_raw)

# plt.xlabel("Count")
# plt.ylabel("frequency")
# plt.title("Counts in a 3s interval")

# plt.plot(x,
#        fit_func(x, *parameters),
#       marker='o',
#         label='fit',
#)


fig.legend(bbox_to_anchor=(0.88,0.95))
fig.tight_layout()
fig.set_size_inches(5,5)
plt.savefig('histogram_new.svg')
plt.show()

print(parameters)
print(cov_matrix)

# https://stackoverflow.com/questions/25828184/fitting-to-poisson-histogram
# https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
# https://matplotlib.org/stable/users/explain/axes/axes_ticks.html
# https://stackoverflow.com/questions/12608788/changing-the-tick-frequency-on-the-x-or-y-axis