Flags = $(GeneralFlags) $(EIGEN3Flags) $(PYBIND11Flags)

.PHONY: all

all: Manifold Simplex Orthogonal Grassmann

Manifold: Manifold.cpp
	$(CXX) -o ../../obj/Manifold.o Manifold.cpp -c $(Flags)

Simplex: Simplex.cpp
	$(CXX) -o ../../obj/Simplex.o Simplex.cpp -c $(Flags)

Orthogonal: Orthogonal.cpp
	$(CXX) -o ../../obj/Orthogonal.o Orthogonal.cpp -c $(Flags)

Grassmann: Grassmann.cpp
	$(CXX) -o ../../obj/Grassmann.o Grassmann.cpp -c $(Flags)
