import matplotlib.pyplot as plt
import numpy as np


n = 1000
xs = np.linspace(0.3700, 1.500, n)

def Cn(l, a, b, c):
    return a + (b/l**2) + (c/l**4)

ri_x = []
ri_y = []
ri_yk = []
is_x = []
is_y = []
is_yk = []

with open('ndataSi') as tsv:
    for line in tsv:
        ri_x.append(float(line.strip().split('\t')[0]))
        ri_y.append(float(line.strip().split('\t')[1]))

with open('kdataSi') as tsv:
    for line in tsv:
        ri_yk.append(float(line.strip().split('\t')[1]))

with open('IndataSi') as tsv:
    for line in tsv:
        is_x.append(float(line.strip().split('\t')[0]) * 0.001)
        is_y.append(float(line.strip().split('\t')[1]))

with open('IkdataSi') as tsv:
    for line in tsv:
        is_yk.append(float(line.strip().split('\t')[1]))


z = np.polyfit(is_x, is_y, 5)
f = np.poly1d(z)

for x1 in xs:
    plt.plot(x1, f(x1), 'b+')

plt.plot(ri_x, ri_y, label='Database n data')
plt.plot(ri_x, ri_yk, label='Database k data')
plt.plot(is_x, is_y, label='Our n data')
plt.plot(is_x, is_yk, label='Our k data')

plt.xlabel = 'wavelength'
plt.ylabel = 'n, k'
plt.legend()

plt.show()




'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


n = 1000
xs = np.linspace(0.3700, 1.500, n)

def Cn(l, a, b, c):
    return a + (b/l**2) + (c/l**4)

ri_x = []
ri_y = []
ri_yk = []
is_x = []
is_y = []
is_yk = []

with open('ndataSi') as tsv:
    for line in tsv:
        ri_x.append(float(line.strip().split('\t')[0]))
        ri_y.append(float(line.strip().split('\t')[1]))

with open('kdataSi') as tsv:
    for line in tsv:
        ri_yk.append(float(line.strip().split('\t')[1]))

with open('IndataSi') as tsv:
    for line in tsv:
        is_x.append(float(line.strip().split('\t')[0]) * 0.001)
        is_y.append(float(line.strip().split('\t')[1]))

with open('IkdataSi') as tsv:
    for line in tsv:
        is_yk.append(float(line.strip().split('\t')[1]))



popt, pcov = curve_fit(Cn, is_x, is_y)
popt2, pcov2 = curve_fit(Cn, ri_x, ri_y)

print('Our Cauchy Coefficients:', popt)
print('Database Cauchy Coefficients', popt2)

plt.plot(xs, Cn(xs, *popt), label='Ours Extrapolated')
plt.plot(xs, Cn(xs, *popt2), label='Database Extrapolated')



print('Our estimated n at 1340:', Cn(1.340, *popt))
print('Database estimated n at 1340:', Cn(1.340, *popt2))


plt.plot(ri_x, ri_y, label='Database n data')
plt.plot(ri_x, ri_yk, label='Database k data')
plt.plot(is_x, is_y, label='Our n data')
plt.plot(is_x, is_yk, label='Our k data')
plt.gcf().set_size_inches(25, 15)

plt.xlabel = 'wavelength'
plt.ylabel = 'n, k'
plt.legend()

plt.show()

'''