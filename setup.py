import os
import urllib.request
import tarfile
import subprocess
from setuptools import setup
from setuptools.command.build import build

class CustomBuild(build):
	def run(self):

		pwd = os.path.dirname(__file__)

		# Checking commands
		MAKE = os.getenv("MAKE", default = "make")
		print("MAKE is %s." % MAKE)
		CXX = os.getenv("CXX", default = "g++")
		print("CXX is %s." % CXX)
		AR = os.getenv("AR", default = "ar")
		print("AR is %s." % AR)

		# Checking dependencies
		PYTHON3 = os.getenv("PYTHON3")
		print("Looking for Python.h at %s ..." % PYTHON3, end='')
		if os.path.isfile(PYTHON3 + "Python.h"):
			print("Found!")
		else:
			raise RuntimeError("Python.h does not exist!")
		EIGEN3 = os.getenv("EIGEN3")
		if len(EIGEN3) > 0:
			print("Looking for Eigen3 at %s ..." % EIGEN3, end='')
			if os.path.exists(EIGEN3 + "/Eigen/") and os.path.exists(EIGEN3 + "/unsupported/") and os.path.isfile(EIGEN3 + "/signature_of_eigen3_matrix_library"):
				print("Found!")
			else:
				raise RuntimeError("Python.h does not exist!")
		else:
			print("The environment variable $EIGEN3 is not set. -> Downloading ...")
			urllib.request.urlretrieve("https://gitlab.com/libeigen/eigen/-/archive/3.4-rc1/eigen-3.4-rc1.tar.gz", pwd)
			with tarfile.open(tarball_path) as tar:
				tar.extractall(path = pwd) # Directory: eigen-3.4-rc1
			EIGEN3 = pwd + "/eigen-3.4-rc1/"
		PYBIND11 = os.popen("pip show pybind11 | grep 'Location:'").split()[1] + "/pybind11/include/"

		# Configuring the makefile
		subprocess.check_call(["sed", "-i", "'%s/__MAKE__/" + MAKE + "/g", "makefile"])
		subprocess.check_call(["sed", "-i", "'%s/__CXX__/" + CXX + "/g", "makefile"])
		subprocess.check_call(["sed", "-i", "'%s/__AR__/" + AR + "/g", "makefile"])
		subprocess.check_call(["sed", "-i", "'%s/__PYTHON3__/" + PYTHON3 + "/g", "makefile"])
		subprocess.check_call(["sed", "-i", "'%s/__EIGEN3__/" + EIGEN3 + "/g", "makefile"])
		subprocess.check_call(["sed", "-i", "'%s/__PYBIND11__/" + PYBIND11 + "/g", "makefile"])

		# Make
		nproc = os.popen("nproc")
		subprocess.check_call(["sed", "-i", "'%s/__OBJ__/__CPP__/g", "makefile"])
		subprocess.check_call([MAKE, "-j", nproc])
		subprocess.check_call(["sed", "-i", "'%s/__CPP__/__PYTHON__/g", "makefile"])
		subprocess.check_call([MAKE, "-j", nproc])

		super().run()


setup(
		name = "Maniverse",
		author = "FreemanTheMaverick",
		description = "Function optimization on manifolds",
		version = "0.2",
		url = "https://github.com/FreemanTheMaverick/Maniverse.git",
		packages = ["src"],
		package_data = { "src": ["lib/*"] },
		cmdclass = { "build": CustomBuild },
		install_requires = ["pybind11>=2.13.6"],
		classifiers = ["Programming Language :: Python :: 3"]
)
