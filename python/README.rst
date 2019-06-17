=====
RFCDE
=====


The ``RFCDE`` package implements random forests for conditional density
estimation.

Installation
------------

RFCDE is availabile from PyPI; to install use \`pip\`

.. code:: shell

    pip install rfcde


Troubleshooting
---------------

* Make sure that `statsmodels` is updated at the latest version - there an issue with version `0.8.0` in which `datetools` is not correctly imported;
* There might be issues installing on Mac OS Mojave, as there are known issues with XCode 10.X (this stack overflow [article](https://stackoverflow.com/questions/52509602/cant-compile-c-program-on-a-mac-after-upgrade-to-mojave) gives a more in-depth explanations).
    If the pip installation does not work, try:
    - Set the global variable `export MACOSX_DEPLOYMENT_TARGET=10.X` (where 10.X is the OS version - for Mojave is 10.14), and then re-run the installation
    - Include `CFLAGS='-stdlib=libstdc++'` before pip install command, so `CFLAGS='-stdlib=libstdc++' pip install rfcde`