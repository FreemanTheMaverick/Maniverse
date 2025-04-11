import os
import wget
import tarfile
import subprocess
from glob import glob
from setuptools import setup, find_packages
from setuptools.command.build import build
import pybind11
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, naive_recompile

__version__ = "0.2.9"
pwd = os.path.dirname(__file__)

# Checking dependencies
PYTHON3 = os.getenv("PYTHON3", default = '')
print("Looking for Python.h at %s ..." % PYTHON3, end='')
if os.path.isfile(PYTHON3 + "/Python.h"):
	print("Found!")
else:
	raise RuntimeError("Python.h does not exist!")
EIGEN3 = os.getenv("EIGEN3", default = '')
if len(EIGEN3) > 0:
	print("Looking for Eigen3 at %s ..." % EIGEN3, end='')
	if os.path.exists(EIGEN3 + "/Eigen/") and os.path.exists(EIGEN3 + "/unsupported/") and os.path.isfile(EIGEN3 + "/signature_of_eigen3_matrix_library"):
		print("Found!")
	else:
		raise RuntimeError("Python.h does not exist!")
else:
	print("The environment variable $EIGEN3 is not set. -> Downloading ...")
	filename = wget.download("https://gitlab.com/libeigen/eigen/-/archive/3.4-rc1/eigen-3.4-rc1.tar.gz", bar = None)
	with tarfile.open(filename) as tar:
		tar.extractall(path = pwd) # Directory: eigen-3.4-rc1
	EIGEN3 = pwd + "/eigen-3.4-rc1/"
	print("EIGEN3 is %s." % EIGEN3)
PYBIND11 = pybind11.get_include()
print("PYBIND11 is %s." % PYBIND11)

BASE_DIR = os.path.dirname(__file__)
os.chdir(BASE_DIR)

ParallelCompile(
	"NPY_NUM_BUILD_JOBS",
	needs_recompile = naive_recompile
).install()

MV_CPP = sorted(glob("src/*.cpp") + glob("src/*/*.cpp"))
MV_HEADER = sorted(glob("src/*.h") + glob("src/*/*.h"))
ext_modules = [ Pybind11Extension(
	"Maniverse",
	MV_CPP,
	undef_macros = ["DEBUG"],
	include_dirs = [PYTHON3, EIGEN3, PYBIND11, MV_HEADER],
	extra_compile_args = ["-O3", "-D__PYTHON__", "-DEIGEN_INITIALIZE_MATRICES_BY_ZERO"],
	cxx_std = 17,
	language = "c++"
)]

setup(
		name = "Maniverse",
		version = __version__,
		author = "FreemanTheMaverick",
		description = "Function optimization on manifolds",
		ext_modules = ext_modules,
		url = "https://github.com/FreemanTheMaverick/Maniverse.git",
		classifiers = ["Programming Language :: Python :: 3"]
)
