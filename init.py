import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import scattr
import numpy as np
import scipy.stats

import corner

data = np.loadtxt('test/tabluca.txt')
data = data[data[:,3]!=0,:]

xc = data[:,0]; xs = data[:,1]
yc = data[:,2]; ys = data[:,3]

x = scattr.data(dist=scattr.dist.Normal(loc=xc,scale=xs),golog=True)
y = scattr.data(dist=scattr.dist.Normal(loc=yc,scale=ys),golog=True)

m = scattr.dist.Uniform(low= 0.00E+00,high=1.00E+02)
c = scattr.dist.Uniform(low=-10.00E+00,high=10.00E+00)

res = scattr.sample(x,y,m,c,nwarmup=5000,nsample=5000)

# ----

corner.corner(np.vstack([res.samples['m'],res.samples['c']]).T,labels=['m','c'])
plt.show(); plt.close()

# ----

xline = np.linspace(0.1,200,100)
yline = 10**(np.log10(xline[:,None])*res.samples['m'][None,:]+res.samples['c'][None,:])
yline = np.array([np.quantile(yi,[0.16,0.50,0.84]) for yi in yline]).T

plt.plot(xline,yline[1])
plt.fill_between(xline,yline[0],yline[2])

plt.errorbar(xc,yc,xerr=xs,yerr=ys,fmt='o',color='k',ecolor='k',elinewidth=1,capsize=0)
plt.xscale('log')
plt.yscale('log')
plt.show(); plt.close()
