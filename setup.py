import os
import urllib.request
import tarfile
from glob import glob
from setuptools import setup
from setuptools.command.build import build
import pybind11
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, naive_recompile

__version__ = "1.0.1"

# Downloading Eigen3
pwd = os.path.dirname(__file__)
EIGEN3 = pwd + "/eigen-3.4-rc1/"
class CustomBuild(build):
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
MV_HEADER = sorted(glob("src/*.h") + glob("src/*/*.h"))
ext_module = Pybind11Extension(
	"Maniverse",
	MV_CPP,
	include_dirs = [EIGEN3],
	depends = MV_HEADER,
	extra_compile_args = ["-O3", "-D__PYTHON__", "-DEIGEN_INITIALIZE_MATRICES_BY_ZERO"],
	language = "c++",
	cxx_std = 17
)

setup(
	name = "Maniverse",
	version = __version__,
	author = "FreemanTheMaverick",
	description = "Numerical optimization on manifolds",
	long_description = open("README.md").read(),
	long_description_content_type = "text/markdown",
	url = "https://github.com/FreemanTheMaverick/Maniverse.git",
	cmdclass = {"build": CustomBuild},
	ext_modules = [ext_module],
	classifiers = ["Programming Language :: Python :: 3"]
)
