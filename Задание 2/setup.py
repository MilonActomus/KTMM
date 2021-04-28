from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("E:/Python/Задание2/A_cyth.pyx"), include_dirs=[numpy.get_include()]
)