import matplotlib.pyplot as plt

import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jp
import jax.scipy

import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import OrderedTransform
from numpyro.contrib.funsor import config_enumerate

import corner
import sys

try:
  import arviz
except ImportError:
  arviz = None
  print('arviz not found, some features will not work')

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
    
    if arviz is not None:
      self.az   = arviz.from_numpyro(self.samp)
      self.loo  = arviz.loo(self.az)
      self.waic = arviz.waic(self.az)
    else:
      self.az   = None
      self.loo  = None
      self.waic = None
      
    for var in ['xk','xs','ys']:
      self.samp._states['z'].pop(var,None)

    self.samp.print_summary()

    self.samples = self.samp.get_samples()


  @staticmethod
  def _sampler(x,y,m,c,
               sx=dist.Uniform(low=0.00,high=1.00E+10),
               sy=dist.Uniform(low=0.00,high=1.00E+10),
               nsample=1000,nwarmup=1000,nchains=1,sampler='nuts',nk=1,pinit=None):

    if not x.scatter: sx = 0.00
    if not y.scatter: sy = 0.00
    
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

      sxi = numpyro.sample('sx',sx) if x.scatter and hasattr(sx,'log_prob') else sx
      syi = numpyro.sample('sy',sy) if y.scatter and hasattr(sy,'log_prob') else sy

      with numpyro.plate('data',x.size):
        zk = numpyro.sample('zk',dist.Categorical(probs=pk)) if nk>1 else 0
        xk = numpyro.sample('xk',dist.Normal(loc=mk.at[zk].get(),scale=jp.sqrt(tk.at[zk].get())))

        xi = numpyro.deterministic('xi',(xk-jp.log10(x.pivot) if x.uselog else xk/x.pivot))
        yi = numpyro.deterministic('yi',xi*mi+ci)

        yk = yi+jp.log10(y.pivot) if y.uselog else yi*y.pivot
        
        xs = numpyro.sample('xs',dist.Normal(loc=xk,scale=sxi)) if x.scatter else xk
        ys = numpyro.sample('ys',dist.Normal(loc=yk,scale=jp.sqrt(syi**2+(mi*sxi)**2))) if y.scatter else yk

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
    samp.run(seed,init_params=pinit)

    setattr(samp,'data',{'x': x,'y': y})
    setattr(samp,'fixed',{})
    for key in ['m','c']:
      if not hasattr(eval(key),'log_prob'): samp.fixed.update({key: eval(key)})

    if x.scatter and not hasattr(sx,'log_prob'): samp.fixed.update({'sx': sx})
    if y.scatter and not hasattr(sy,'log_prob'): samp.fixed.update({'sy': sy})

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

    return corner.corner(np.vstack(samples).T,labels=labels,**kwargs)


  def gen_model(self,fig=None,xl=0.02,**kwargs):    
    nsamp = self.samples[list(self.samples.keys())[0]].shape[0]

    pars = {}
    for key in ['m','c','sx','sy']:
      if key in self.samples.keys():
        pars[key] = np.asarray(self.samples[key])
      elif key in self.samp.fixed.keys():
        pars[key] = np.full(nsamp,self.samp.fixed[key])
    
    x = self.samp.data['x']
    y = self.samp.data['y']

    if isinstance(xl,(int,float)):
      xline = np.linspace(np.power(10,np.log10(x.loc.min()-x.scale[np.argmin(x.loc)])-xl),
                          np.power(10,np.log10(x.loc.max()+x.scale[np.argmax(x.loc)])+xl),100)

    yline = np.log10(xline) if x.uselog else xline
    yline = (yline-np.log10(x.pivot) if x.uselog else yline/x.pivot)
    yline = yline[:,None]*pars['m'][None,:]+pars['c'][None,:]

    yline = yline+np.log10(y.pivot) if y.uselog else yline*y.pivot
    yline = 10**yline if y.uselog else yline

    yline = np.array([np.quantile(yi,[0.16,0.50,0.84]) for yi in yline]).T

    return xline, yline