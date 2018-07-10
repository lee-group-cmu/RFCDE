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

Citation
===

```text
@article{pospisil2018rfcde,
  title={RFCDE: Random Forests for Conditional Density Estimation},
  author={Pospisil, Taylor and Lee, Ann B},
  journal={arXiv preprint arXiv:1804.05753},
  year={2018}
}
```