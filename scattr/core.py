import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import OrderedTransform

# Build data structure
# --------------------------------------------------------------------------------
class data:
  def __init__(self,loc,scale,uselog=False,scatter=False):
    self.dist   = dist.Normal(loc=loc,scale=scale)
  
    self.uselog  = uselog
    self.scatter = scatter

    self.pivot   = 10**jp.median(jp.log10(self.dist.loc)) if uselog else jp.median(self.dist.loc)

    self.size   = self.dist.loc.size

    if self.uselog:
      self.min = jp.log10(self.dist.loc.min())-(jp.log10(self.dist.loc.max())-jp.log10(self.dist.loc.min()))
      self.max = jp.log10(self.dist.loc.max())+(jp.log10(self.dist.loc.max())-jp.log10(self.dist.loc.min()))
    else:
      self.min = self.dist.loc.min()-10*self.dist.scale[jp.argmin(self.dist.loc)]
      self.max = self.dist.loc.max()+10*self.dist.scale[jp.argmax(self.dist.loc)] 

# --------------------------------------------------------------------------------
class sample:
  def __init__(self,x,y,m,c,nsample=1000,nwarmup=1000,**kwargs):

    nk = kwargs.get('nk',1)

    def model():
      mi = numpyro.sample('m',m)
      ci = numpyro.sample('c',c)

      ws = numpyro.sample('ws',dist.Uniform(low=0.00E+00,high=1.00E+10))
      ms = numpyro.sample('ms',dist.Uniform(low=x.min,high=x.max))
      us = numpyro.sample('us',dist.InverseGamma(concentration=5.00E-01,rate=5.00E-01*ws))

      with numpyro.plate('mode',nk):
        tk = numpyro.sample('tk',dist.InverseGamma(concentration=5.00E-01,rate=5.00E-01*ws))
        mk = numpyro.sample('mk',dist.TransformedDistribution(dist.Normal(loc=ms,scale=jp.sqrt(us)).expand([nk]),OrderedTransform()))

      if nk>1: pk = numpyro.sample('pk',dist.Dirichlet(jp.ones(nk)))

      sx = numpyro.sample('sx',dist.Uniform(low=0.00,high=1.00E+10)) if x.scatter else 0.00
      sy = numpyro.sample('sy',dist.Uniform(low=0.00,high=1.00E+10)) if y.scatter else 0.00

      with numpyro.plate('data',x.size):
        zk = numpyro.sample('zk',dist.Categorical(probs=pk)) if nk>1 else 0
        xk = numpyro.sample('xk',dist.Normal(loc=mk[zk],scale=jp.sqrt(tk[zk])))

        xi = numpyro.deterministic('xi',(xk-jp.log10(x.pivot) if x.uselog else xk/x.pivot))
        yi = numpyro.deterministic('yi',xi*mi+ci)

        yk = yi+jp.log10(y.pivot) if y.uselog else yi*y.pivot
        
        xs = numpyro.sample('xs',dist.Normal(loc=xk,scale=sx)) if x.scatter else xk
      # ys = numpyro.sample('ys',dist.Normal(loc=yk,scale=sy)) if y.scatter else yk
        ys = numpyro.sample('ys',dist.Normal(loc=yk,scale=jp.sqrt(sy**2+(mi*sx)**2))) if y.scatter else yk

        xobs = 10**xs if x.uselog else xs
        yobs = 10**ys if y.uselog else ys

        numpyro.sample('xobs',x.dist,obs=xobs)
        numpyro.sample('yobs',y.dist,obs=yobs)
    
    rkey = jax.random.PRNGKey(0) 
    rkey, seed = jax.random.split(rkey)

    self.kern = numpyro.infer.NUTS(model)
    self.mcmc = numpyro.infer.MCMC(self.kern,num_warmup=nwarmup,num_samples=nsample)
    self.mcmc.run(seed)

    for var in ['xk','xs','ys']:
      self.mcmc._states['z'].pop(var,None)

    self.mcmc.print_summary()

    self.samples = self.mcmc.get_samples()

