import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jp

import numpyro
import numpyro.distributions as dist

class data:
  def __init__(self,dist,golog=False):
    self.dist = dist
    self.golog = golog

class sample:
  def __init__(self,x,y,m,c,nsample=1000,nwarmup=1000):
    xget = lambda d: jp.log10(d) if x.golog else d

    xmin = xget(x.dist.loc.min())-10*xget(x.dist.scale[jp.argmin(x.dist.loc)])
    xmax = xget(x.dist.loc.max())+10*xget(x.dist.scale[jp.argmax(x.dist.loc)])    

    if x.golog:
      xmin = jp.log10(x.dist.loc.min())-(jp.log10(x.dist.loc.max())-jp.log10(x.dist.loc.min()))
      xmax = jp.log10(x.dist.loc.max())+(jp.log10(x.dist.loc.max())-jp.log10(x.dist.loc.min()))
    else:
      xmin = x.dist.loc.min()-10*x.dist.scale[jp.argmin(x.dist.loc)]
      xmax = x.dist.loc.max()+10*x.dist.scale[jp.argmax(x.dist.loc)] 

    print(xmin,xmax,jp.log10(x.dist.loc.min()),jp.log10(x.dist.loc.max()))
    xnum = x.dist.loc.size

    def model():
      mini = numpyro.sample('m',m)
      cini = numpyro.sample('c',c)

      with numpyro.plate('data',xnum):
        xini = numpyro.sample('x',dist.Uniform(low=xmin,high=xmax))
        yini = numpyro.deterministic('y',xini*mini+cini)

        xout = 10**xini if x.golog else xini
        yout = 10**yini if y.golog else yini

        numpyro.sample('xobs',x.dist,obs=xout)
        numpyro.sample('yobs',y.dist,obs=yout)
    
    rkey = jax.random.PRNGKey(0) 
    rkey, seed = jax.random.split(rkey)

    self.kern = numpyro.infer.NUTS(model)
    self.mcmc = numpyro.infer.MCMC(self.kern,num_warmup=nwarmup,num_samples=nsample)
    self.mcmc.run(seed)

    self.mcmc.print_summary()

    self.samples = self.mcmc.get_samples()

