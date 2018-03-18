from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

import numpy as np

with open("README.rst", "r") as f:
    README_TEXT = f.read()

ext_modules = cythonize([
          Extension("rfcde.ForestWrapper",
                    sources = ['src/rfcde/ForestWrapper.pyx',
                               'src/rfcde/Forest.cpp',
                               'src/rfcde/Tree.cpp',
                               'src/rfcde/Node.cpp',
                               'src/rfcde/Split.cpp',
                               'src/rfcde/helpers.cpp'],
                    libraries = ['src/rfcde/Forest.h'
                                 'src/rfcde/Tree.h',
                                 'src/rfcde/Node.h',
                                 'src/rfcde/Split.h',
                                 'src/rfcde/helpers.h'],
                    extra_compile_args = ['-std=c++11'],
                    include_dirs = [np.get_include()],
                    language='c++')])

for e in ext_modules:
    e.cython_directives = {"embedsignature": True}

setup(name="rfcde",
      version="0.1.1",
      license="MIT",
      description="Fits random forest conditional density estimate using conditional density loss for splits.",
      long_description = README_TEXT,
      author           = "Taylor Pospisil",
      author_email     = "tpospisi@andrew.cmu.edu",
      maintainer       = "tpospisi@andrew.cmu.edu",
      url="https://github.com/tpospisi/rfcde/python",
      classifiers = ["License :: OSI Approved :: MIT License",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence",
                     "Programming Language :: Python :: 2.7",
                     "Programming Language :: Python :: 3.6",
                     "Programming Language :: Cython",
                     "Programming Language :: C++"],
      keywords = ["conditional density estimation", "random forests"],
      package_dir={"": "src"},
      packages=["rfcde"],
      python_requires=">=2.7",
      install_requires=["numpy", "cython", "scipy"],
      setup_requires=["cython", "pytest-runner"],
      tests_require=["pytest"],
      zip_safe=False,
      include_package_data=True,
      ext_modules = ext_modules,
)
