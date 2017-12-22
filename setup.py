from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

ext = Extension("py_scvb0",
                sources=["py_scvb0.pyx", "lda_scvb0_thread.c"],
                )

setup(name = 'py_scvb0', ext_modules = cythonize([ext]))

