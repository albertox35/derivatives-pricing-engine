"""Build del nucleo nativo.

El flag de OpenMP se aniade de forma condicional: si el compilador no lo
soporta, la extension compila igual y el nucleo corre en un solo hilo. Un build
que falla por una optimizacion opcional es peor que un build mas lento.
"""

from __future__ import annotations

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++20", "/openmp"]
    extra_link_args: list[str] = []
else:
    extra_compile_args = ["-O3", "-std=c++20", "-fopenmp", "-ffp-contract=off"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "pricing._kernel",
        ["cpp/kernel.cpp"],
        cxx_std=20,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
