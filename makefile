#################################################
# Change these for your computer configuration. #
#################################################

export MAKE = __MAKE__
export CXX = __CXX__
export AR = __AR__
export OBJ = __OBJ__
# $(OBJ) should be either "__CPP__" or "__PYTHON__".
# The two versions can coexist.
# You may compile twice, once with "__CPP__" and the other with "__PYTHON__", to obtain both versions.
# In the "lib/" directory, "libmaniverse.a" and "libmaniverse.so" are for CPP and "Maniverse.so" is for Python.

# If $(OBJ) is "__PYTHON__", you need to set $(PYTHON3) and $(PYBIND11).
export PYTHON3 = __PYTHON3__
# The path where you can find "Python.h".
export PYBIND11 = __PYBIND11__
# The path where you can find "pybind11/attr.h", "pybind11/stl/", etc.
export EIGEN3 = __EIGEN3__
# The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".


#######################################################
# The following codes are not supposed to be changed. #
#######################################################

export GeneralFlags = -Wall -Wextra -Wpedantic -O3 -std=c++17 -fPIC -D$(OBJ)
export PYBIND11Flags = -isystem $(PYTHON3) -isystem $(PYBIND11)
export EIGEN3Flags = -isystem $(EIGEN3) -march=native -DEIGEN_INITIALIZE_MATRICES_BY_ZERO
export Flags = $(GeneralFlags) $(PYBIND11Flags) $(EIGEN3Flags)

.PHONY: all

all:
	mkdir -p obj/ lib/ include/
	$(MAKE) -C src/
	if [ $(OBJ) == __CPP__ ]; then\
		$(AR) -rv lib/libmaniverse.a obj/__CPP__*.o;\
		$(CXX) -shared -o lib/libmaniverse.so obj/__CPP__*.o;\
	elif [ $(OBJ) == __PYTHON__ ]; then\
		$(CXX) -shared -o lib/Maniverse.so obj/__PYTHON__*.o;\
	fi
	mkdir -p include/Maniverse/
	mkdir -p include/Maniverse/Manifold/
	mkdir -p include/Maniverse/Optimizer/
	find src/Manifold/ -type f -name "*h" ! -name "Py*" -exec cp {} include/Maniverse/Manifold/ \;
	find src/Optimizer/ -type f -name "*h" ! -name "Py*" -exec cp {} include/Maniverse/Optimizer/ \;
	find include/Maniverse/Manifold/ -type f -name "*h" -exec sed -i "s/EigenMatrix/Eigen::MatrixXd/g" {} \;
	find include/Maniverse/Optimizer/ -type f -name "*h" -exec sed -i "s/EigenMatrix/Eigen::MatrixXd/g" {} \;

clean:
	rm -rf obj/ lib/ include/
