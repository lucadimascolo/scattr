import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jp
import jax.scipy

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import OrderedTransform

import arviz
import sys

# from numpyro.contrib.nested_sampling import NestedSampler

# Sawicki upper limits
# --------------------------------------------------------------------------------
def sawicki(value,loc,scale):
  perf = (loc-value)/jp.sqrt(2.00*scale**2)
  perf = jax.scipy.special.erf(perf)
  prob = jp.log(jp.sqrt(0.50*jp.pi*scale**2)*(1.00+perf))
  prob -= 0.50*jp.log(2.00*jp.pi*scale**2)
  return jp.where(perf==-1,-sys.float_info.max,prob)
  
# Mixed distribution
# --------------------------------------------------------------------------------
class NormalUpper(dist.Distribution):
    support = dist.constraints.real_vector

    def __init__(self,loc,scale,isupp):
      self.loc, self.scale, self.isupp = loc, scale, isupp
      super().__init__(batch_shape=jp.shape(self.loc), event_shape=())

    def sample(self, key, sample_shape=()):
      raise NotImplementedError

    def log_prob(self, value):      
      prob1 = jax.scipy.stats.norm.logpdf(x = value.at[self.isupp==False].get(),
                                        loc = self.loc.at[self.isupp==False].get(),
                                      scale = self.scale.at[self.isupp==False].get())
            
      prob2 = sawicki(value = value.at[self.isupp==True].get(),
                        loc = self.loc.at[self.isupp==True].get(),
                      scale = self.loc.at[self.isupp==True].get())

      logprob = jp.zeros(value.shape)
      logprob = logprob.at[self.isupp==False].set(prob1)
      logprob = logprob.at[self.isupp==True].set(prob2)
      return logprob


# Build data structure
# --------------------------------------------------------------------------------
class data:
  def __init__(self,loc,scale,uselog=False,scatter=False,isupp=None):
    if isupp is None:
      isupp = jp.zeros(loc.shape,dtype=bool)

    if jp.any(isupp):
      self.dist  = NormalUpper(loc=loc,scale=scale,isupp=isupp)
    else:
      self.dist  = dist.Normal(loc=loc.at[isupp==False].get(),
                           scale=scale.at[isupp==False].get())

    self.uselog  = uselog
    self.scatter = scatter

    self.obs     = loc.at[isupp==False].get()

    self.pivot   = 10**jp.median(jp.log10(self.obs)) if uselog else jp.median(self.obs)

    self.size    = loc.shape[0]

    if self.uselog:
      self.min = jp.log10(self.obs.min())-(jp.log10(self.obs.max())-jp.log10(self.obs.min()))
      self.max = jp.log10(self.obs.max())+(jp.log10(self.obs.max())-jp.log10(self.obs.min()))
    else:
      self.min = self.obs.min()-10*self.dist.scale[jp.argmin(self.obs)]
      self.max = self.obs.max()+10*self.dist.scale[jp.argmax(self.obs)] 

# --------------------------------------------------------------------------------
class sample:
  def __init__(self,x,y,m,c,nsample=1000,nwarmup=1000,sampler='nuts',**kwargs):

    nk = kwargs.get('nk',1)

    def model():
      mi = numpyro.sample('m',m) if hasattr(m,'log_prob') else m
      ci = numpyro.sample('c',c) if hasattr(c,'log_prob') else c

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
        ys = numpyro.sample('ys',dist.Normal(loc=yk,scale=jp.sqrt(sy**2+(mi*sx)**2))) if y.scatter else yk

        xobs = 10**xs if x.uselog else xs
        yobs = 10**ys if y.uselog else ys

        numpyro.factor('obs',x.dist.log_prob(xobs)+y.dist.log_prob(yobs))

    rkey = jax.random.PRNGKey(0) 
    rkey, seed = jax.random.split(rkey)

    if sampler=='nuts':
      self.kern = numpyro.infer.NUTS(model)
      self.samp = numpyro.infer.MCMC(self.kern,num_warmup=nwarmup,num_samples=nsample)
    elif sampler=='nested':
      raise NotImplementedError
 
    self.samp.run(seed)
    
    self.az   = arviz.from_numpyro(self.samp)
    self.loo  = arviz.loo(self.az)
    self.waic = arviz.waic(self.az)

    for var in ['xk','xs','ys']:
      self.samp._states['z'].pop(var,None)

    self.samp.print_summary()

    print(self.loo)
    print(self.waic)

    self.samples = self.samp.get_samples()
