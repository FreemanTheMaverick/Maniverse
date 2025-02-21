export MAKE = make
export CXX = g++
export AR = ar

export PYTHON3 = /home/yzhangnn/scratch/anaconda3/include/python3.11
# The path where you can find "Python.h".
export PYBIND11 = /home/yzhangnn/pybind11/include/
# The path where you can find "pybind11/".
export EIGEN3 = /home/yzhangnn/eigen/
# The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".

export GeneralFlags = -Wall -Wextra -Wpedantic -O3 -std=c++17 -fPIC
export PYBIND11Flags = -isystem $(PYTHON3) -isystem $(PYBIND11)
export EIGEN3Flags = -isystem $(EIGEN3) -march=native -DEIGEN_INITIALIZE_MATRICES_BY_ZERO
export Flags = $(GeneralFlags) $(PYBIND11Flags) $(EIGEN3Flags)

.PHONY: all

all:
	mkdir -p obj/ lib/ include/
	$(MAKE) -C src/
	$(AR) -rv lib/libmaniverse.a obj/*.o
	$(CXX) -shared -o lib/libmaniverse.so obj/*.o
	$(CXX) -shared -o lib/Maniverse.so obj/*.o
	mkdir -p include/Maniverse/
	mkdir -p include/Maniverse/Manifold/
	mkdir -p include/Maniverse/Optimizer/
	find src/Manifold/ -type f -name "*h" ! -name "Py*" -exec cp {} include/Maniverse/Manifold/ \;
	find src/Optimizer/ -type f -name "*h" ! -name "Py*" -exec cp {} include/Maniverse/Optimizer/ \;
	find include/Maniverse/Manifold/ -type f -name "*h" -exec sed -i "s/EigenMatrix/Eigen::MatrixXd/g" {} \;
	find include/Maniverse/Optimizer/ -type f -name "*h" -exec sed -i "s/EigenMatrix/Eigen::MatrixXd/g" {} \;

clean:
	rm -rf obj/ lib/ include/
