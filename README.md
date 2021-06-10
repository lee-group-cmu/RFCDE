RFCDE
===

This repository provides an implementation of random forests designed
for conditional density estimation (https://arxiv.org/abs/1804.05753).
R and python packages are available. For installation details and
package-specific documentation see the subdirectories _r_ and
_python_. Both languages use a common C++ library which can be found
in the _cpp_ subdirectory.


Photo-Z Example
===

We apply RFCDE to photometric redshift estimation for the LSST DESC
DC-1. For members of the LSST DESC, you can find information on
obtaining the data
[here](https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=LSSTDESC&title=DC1+resources).

```python
import numpy as np
import pandas as pd
import rfcde

# Read in data
def process_data(feature_file, has_z=False):
    """Processes buzzard data"""
    df = pd.read_table(feature_file, sep=" ")
    df["ug"] = df["u"] - df["g"]

    df.assign(ug = df.u - df.g,
              gr = df.g - df.r,
              ri = df.r - df.i,
              iz = df.i - df.z,
              zy = df.z - df.y,
              ug_err = np.sqrt(df['u.err'] ** 2 + df['g.err'] ** 2),
              gr_err = np.sqrt(df['g.err'] ** 2 + df['r.err'] ** 2),
              ri_err = np.sqrt(df['r.err'] ** 2 + df['i.err'] ** 2),
              iz_err = np.sqrt(df['i.err'] ** 2 + df['z.err'] ** 2),
              zy_err = np.sqrt(df['z.err'] ** 2 + df['y.err'] ** 2))

    if has_z:
        z = df.redshift.as_matrix()
        df.drop('redshift', axis=1, inplace=True)
    else:
        z = None

    return df.as_matrix(), z

x_train, z_train = process_data('buzzard_spec_witherrors_mass.txt', has_z=True)
x_test, _ = process_data('buzzard_phot_witherrors_mass.txt', has_z=False)

# Fit the model
n_trees = 1000
mtry = 4
node_size = 20
n_basis = 31

forest = rfcde.RFCDE(n_trees=n_trees, mtry=mtry, node_size=node_size, n_basis=n_basis)
forest.train(x_train, z_train)

# Make predictions
bandwidth = 0.005
n_grid = 200
z_grid = np.linspace(0, 2, n_grid)
density = forest.predict(x_test, z_grid, bandwidth)
```

fRFCDE
===

Functional RFCDE (fRFCDE) is a variant of RFCDE which can efficiently handle functional input (https://arxiv.org/abs/1906.07177). In this variant, functional covariates are grouped together according to a Poisson process with parameter <sub>&lambda;</sub>.
It is included within the _r_ and _python_ package and the parameter <sub>&lambda;</sub> can be set as follows:

```python
import numpy as np
import rfcde

# Parameters
n_trees = 1000     # Number of trees in the forest
mtry = 4           # Number of variables to potentially split at in each node
node_size = 20     # Smallest node size
n_basis = 15       # Number of basis functions
bandwidth = 0.2    # Kernel bandwith - used for prediction only
lambda_param = 10  # Poisson Process parameter

# Fit the model
functional_forest = rfcde.RFCDE(n_trees=n_trees, mtry=mtry, node_size=node_size, 
                                n_basis=n_basis)
functional_forest.train(x_train, y_train, flamba=lambda_param)

# ... Same as RFCDE for prediction ...
```

```R
library(RFCDE)

# Parameters
n_trees <- 1000     # Number of trees in the forest
mtry <- 4           # Number of variables to potentially split at in each node
node_size <- 20     # Smallest node size
n_basis <- 15       # Number of basis functions
bandwidth <- 0.2    # Kernel bandwith - used for prediction only
lambda_param <- 10  # Poisson Process parameter

# Fit the model
functional_forest <- RFCDE::RFCDE(x_train, y_train, n_trees = n_trees, mtry = mtry, 
                                  node_size = node_size, n_basis = n_basis, 
                                  flambda = lambda_param)

# ... Same as RFCDE for prediction ...
```

Citation
===

```text
@article{pospisil2018rfcde,
  title={RFCDE: Random Forests for Conditional Density Estimation},
  author={Pospisil, Taylor and Lee, Ann B},
  journal={arXiv preprint arXiv:1804.05753},
  year={2018}
}
@article{pospisil2019(f)rfcde,
title={(f)RFCDE: Random Forests for Conditional Density Estimation and Functional Data},
author={Pospisil, Taylor and Lee, Ann B},
journal={arXiv preprint arXiv:1906.07177},
year={2019}
}
@article{dalmasso2020cdetools,
       author = {{Dalmasso}, N. and {Pospisil}, T. and {Lee}, A.~B. and {Izbicki}, R. and
         {Freeman}, P.~E. and {Malz}, A.~I.},
        title = "{Conditional density estimation tools in python and R with applications to photometric redshifts and likelihood-free cosmological inference}",
      journal = {Astronomy and Computing},
         year = 2020,
        month = jan,
       volume = {30},
          eid = {100362},
        pages = {100362},
          doi = {10.1016/j.ascom.2019.100362}
}
```

Troubleshooting (Python)
==

1. Make sure that `statsmodels` is updated at the latest version - there an issue with version `0.8.0` in which `datetools` is not correctly imported;
2. There might be issues installing on Mac OS Mojave, as there are known issues with XCode 10.X (this Stack Overflow [article](https://stackoverflow.com/questions/52509602/cant-compile-c-program-on-a-mac-after-upgrade-to-mojave) gives a more in-depth explanations).
    If the pip installation does not work, try 
    * Set the global variable `export MACOSX_DEPLOYMENT_TARGET=10.X` (where 10.X is the OS version - for Mojave is 10.14), and then re-run the installation
    * Include `CFLAGS='-stdlib=libstdc++'` before pip install command, so `CFLAGS='-stdlib=libstdc++' pip install rfcde`
3. If installing on Mac OS Catalina with Python 3.8, Apple runs Python with `-arch arm64`, which makes the C++ code failing to install. One should run `export ARCHFLAGS="-arch x86_64"` first to setup the `-arch` flag correctly. (see [this issue](https://github.com/giampaolo/psutil/issues/1832)).
