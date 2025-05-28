import os
import urllib.request
import tarfile
from glob import glob
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import pybind11
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, naive_recompile

__version__ = "0.3.3"

# Downloading Eigen3
pwd = os.path.dirname(__file__)
EIGEN3 = pwd + "/eigen-3.4-rc1/"
class CustomBuild(build_py):
	def run(self):
		url = "https://gitlab.com/libeigen/eigen/-/archive/3.4-rc1/eigen-3.4-rc1.tar.gz"
		dest = pwd + "/eigen-3.4-rc1.tar.gz"
		print("Downloading Eigen3 from %s to %s ..." % (url, dest))
		urllib.request.urlretrieve(url, dest)
		print("Extracting %s to %s ..." % (dest, EIGEN3))
		with tarfile.open(dest) as tar:
			tar.extractall(path = pwd) # Directory: eigen-3.4-rc1
		super().run()

ParallelCompile(
	"NPY_NUM_BUILD_JOBS",
	needs_recompile = naive_recompile
).install()

MV_CPP = sorted(glob("src/*.cpp") + glob("src/*/*.cpp"))
ext_modules = [ Pybind11Extension(
	"Maniverse",
	MV_CPP,
	undef_macros = ["DEBUG"],
	include_dirs = [EIGEN3],
	extra_compile_args = ["-O3", "-D__PYTHON__", "-DEIGEN_INITIALIZE_MATRICES_BY_ZERO"],
	cxx_std = 17,
	language = "c++"
)]

setup(
		name = "Maniverse",
		version = __version__,
		author = "FreemanTheMaverick",
		description = "Numerical optimization on manifolds",
		long_description = open("README.md").read(),
		long_description_content_type = "text/markdown",
		url = "https://github.com/FreemanTheMaverick/Maniverse.git",
		cmdclass = {"build_py": CustomBuild},
		ext_modules = ext_modules,
		packages = ["src", "src/Manifold", "src/Optimizer"],
		package_data = {
			"src": ["*.h"],
			"src/Manifold": ["*.h"],
			"src/Optimizer": ["*.h"],
		},
		classifiers = ["Programming Language :: Python :: 3"]
)
