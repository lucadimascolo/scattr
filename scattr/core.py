import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jp
import jax.scipy

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import OrderedTransform
from numpyro.contrib.funsor import config_enumerate

import corner
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
    if not isinstance(loc,  jp.ndarray): loc   = jp.array(loc)
    if not isinstance(scale,jp.ndarray): scale = jp.array(scale)

    self._loc = loc
    self._scale = scale

    self._uselog = uselog
    self.scatter = scatter
    self.isupp = isupp if isupp is not None else jp.zeros(loc.shape, dtype=bool)

    self._update_attributes()

  @property
  def isupp(self):
    return self._isupp

  @isupp.setter
  def isupp(self,value):
    self._isupp = value
    self._update_attributes()

  @property
  def loc(self):
    return self._loc
  
  @loc.setter
  def loc(self,value):
    self._loc = value
    self._update_attributes()
  
  @property
  def scale(self):
    return self._scale
  
  @scale.setter
  def scale(self,value):
    self._scale = value
    self._update_attributes()
  
  @property
  def uselog(self):
    return self._uselog
  
  @uselog.setter
  def uselog(self,value):
    self._uselog = value
    self._update_attributes()

  def _update_attributes(self):
    if jp.any(self._isupp):
      self.dist = NormalUpper(loc=self._loc,scale=self._scale,isupp=self._isupp)
    else:
      self.dist = dist.Normal(loc=self._loc.at[self._isupp==False].get(),
                  scale=self._scale.at[self._isupp==False].get())

    self.obs = self._loc.at[self._isupp==False].get()
    self.pivot = 10**jp.median(jp.log10(self.obs)) if self._uselog else jp.median(self.obs)
    self.size = self._loc.shape[0]

    if self._uselog:
      self.min = jp.log10(self.obs.min())-(jp.log10(self.obs.max())-jp.log10(self.obs.min()))
      self.max = jp.log10(self.obs.max())+(jp.log10(self.obs.max())-jp.log10(self.obs.min()))
    else:
      self.min = self.obs.min() - 10 * self.dist.scale[jp.argmin(self.obs)]
      self.max = self.obs.max() + 10 * self.dist.scale[jp.argmax(self.obs)]


# --------------------------------------------------------------------------------
class sample:
  def __init__(self,method='onestep',**kwargs):

    if method=='multistep':
      ktmp = kwargs.copy()
      
      for key in ['x','y']:
        ktmp[key] = data(loc = ktmp[key].loc[~ktmp[key].isupp],
                       scale = ktmp[key].scale[~ktmp[key].isupp],
                      uselog = ktmp[key].uselog,
                     scatter = ktmp[key].scatter,
                       isupp = None)
      
      samp = self._sampler(**ktmp)
      pars = samp.get_samples()

      kwargs['pinit'] = {}
      for key in ['m','c','sx','sy']:
        kwargs['pinit'][key] = jp.quantile(pars[key],0.50) if key in pars.keys() else kwargs[key]
    
    elif method=='onestep':
      kwargs['pinit'] = None
    
    self.samp = self._sampler(**kwargs)
    
    self.az   = arviz.from_numpyro(self.samp)
    self.loo  = arviz.loo(self.az)
    self.waic = arviz.waic(self.az)

    for var in ['xk','xs','ys']:
      self.samp._states['z'].pop(var,None)

    self.samp.print_summary()

    self.samples = self.samp.get_samples()

  @staticmethod
  def _sampler(x,y,m,c,nsample=1000,nwarmup=1000,nchains=1,sampler='nuts',nk=1,pinit=None):
    
    @config_enumerate
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
        xk = numpyro.sample('xk',dist.Normal(loc=mk.at[zk].get(),scale=jp.sqrt(tk.at[zk].get())))

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
      kern = numpyro.infer.NUTS(model)
      samp = numpyro.infer.MCMC(kern,num_warmup=nwarmup,num_samples=nsample,num_chains=nchains)
    elif sampler=='nested':
      raise NotImplementedError
    samp.run(seed)

    if nk>1:
      post = samp.get_samples()

      predictive = numpyro.infer.Predictive(model,post,infer_discrete=True)
      discrete = predictive(rkey)

      chain_discrete = jax.tree.map(lambda x: x.reshape((nchains,nsample)+x.shape[1:]),discrete)
      samp.get_samples().update(discrete)
      samp.get_samples(group_by_chain=True).update(chain_discrete)

    return samp
  
  def corner(self,parkeys=['m','c','sx','sy'],**kwargs):
    labels, samples = [], []
    for key in parkeys:
      if key in self.samples.keys():
        samples.append(self.samples[key])
        labels.append(key)

    return corner.corner(jp.vstack(samples).T,labels=labels,**kwargs)

  def gen_model(self):
    pass