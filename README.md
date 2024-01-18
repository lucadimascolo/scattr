# scattr

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10528327.svg)](https://doi.org/10.5281/zenodo.10528327)

Yet another linear regression library.

`scattr' is a plug-n-play tool for performing linear regression in the presence of intrinsic scatter and for measurements with uncertainties on both the dependent and independent variables.

This is based on other publicly available tools (specifically, [linmix](https://linmix.readthedocs.io/en/latest/src/linmix.html) and [LIRA](https://cran.r-project.org/web/packages/lira/index.html)), but provides a more statistically consistent approach for dealing with observational uncertainties. More details will be provided asap.

## Installation

To get `scattr` running on your computer, it should be enough to run

```
python -m pip install git+https://github.com/lucadimascolo/scattr.git
```

This will  download and install the latest version of `scattr` as well as all the required dependencies.

## Usage

A preliminary example can be found in `init.py`.
