from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "utils",
        ["src/utils.cpp"],
    ),
]

setup(
    name="utils",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
