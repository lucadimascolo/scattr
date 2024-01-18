# scattr

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10528327.svg)](https://doi.org/10.5281/zenodo.10528327)

_Yet another linear regression library._

`scattr` is a plug-n-play tool for performing linear regression in the presence of intrinsic scatter and for measurements with uncertainties on both the dependent and independent variables.
This is built upon other publicly available tools (specifically, [linmix](https://linmix.readthedocs.io/en/latest/src/linmix.html)<sup>1</sup> and [LIRA](https://cran.r-project.org/web/packages/lira/index.html)<sup>2</sup>), but provides a more statistically consistent approach for dealing with observational uncertainties. More details will be provided asap.

## Installation

To get `scattr` running on your computer, it should be enough to run

```
python -m pip install git+https://github.com/lucadimascolo/scattr.git
```

This will  download and install the latest version of `scattr` as well as all the required dependencies.

## Usage

A preliminary example can be found in `init.py`.

## References

<sup>1</sup> "[_Some Aspects of Measurement Error in Linear Regression of Astronomical Data_](https://ui.adsabs.harvard.edu/abs/2007ApJ...665.1489K/abstract)", Brandon C. Kelly, ApJ, 665, 1489 (2007)

<sup>2</sup> "[_A Bayesian approach to linear regression in astronomy_](https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.2149S/abstract)", Mauro Sereno, MNRAS, 455, 2149 (2016)
