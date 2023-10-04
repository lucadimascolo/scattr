import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jp

import numpyro
import numpyro.distributions as dist

class :
  def __init__(self,x,y,xerr=None,yerr=None):
    self.x = jp.array(x); self.xerr = jp.array(xerr)
    self.y = jp.array(y); self.yerr = jp.array(yerr)

    def model():
      xini = numpyro.sample('xi',dist.___)
      yini = numpyro.deterministic('yi',)

      ysig = 
      xsig = 

      with numpyro.plate('',):
        numpyro.sample('yobs',dist.Normal(,scale=self.yerr),yobs=self.y)