import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jp

import numpyro
import numpyro.distributions as dist

class data:
  def __init__(self,dist,golog):
    self.dist = dist
    self.golog = golog

class sampler:
  def __init__(self,x,y):
    self.x = x
    self.y = y

  def fit(self,a,b,nsample=1000,nwarmup=1000):
    xget = lambda x: jp.log10(x) if self.x.golog else x

    xmin = xget(self.x.dist.loc.min())-10*xget(self.x.dist.scale[jp.argmin(self.x.dist.loc)])
    xmax = xget(self.x.dist.loc.max())+10*xget(self.x.dist.scale[jp.argmin(self.x.dist.loc)])    

    def model():
      aini = numpyro.sample('a',a)
      bini = numpyro.sample('b',b)

      xini = numpyro.sample('x',dist.Uniform(low=xmin,high=xmax))
      yini = numpyro.deterministic('y',xini*aini+bini)

      xout = 10**xini if self.x.golog else xini
      yout = 10**yini if self.y.golog else yini

      numpyro.sample('xobs',self.x.dist,obs=xout)
      numpyro.sample('yobs',self.y.dist,obs=yout)
    
    rkey = jax.random.PRNGKey(0) 
    rkey, seed = jax.random.split(rkey)

    self.kern = numpyro.infer.NUTS(model)
    self.mcmc = numpyro.infer.MCMC(self.kern,num_warmup=nwarmup,num_samples=nsample)
    self.mcmc.run(seed)

    self.samples = self.mcmc.get_samples()