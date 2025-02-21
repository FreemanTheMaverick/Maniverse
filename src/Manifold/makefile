.PHONY: all

all: Manifold Simplex Orthogonal Grassmann TransRotInvPointCloud

Manifold: Manifold.cpp
	$(CXX) -o ../../obj/$(LANG)Manifold.o Manifold.cpp -c $(Flags)

Simplex: Simplex.cpp
	$(CXX) -o ../../obj/$(LANG)Simplex.o Simplex.cpp -c $(Flags)

Orthogonal: Orthogonal.cpp
	$(CXX) -o ../../obj/$(LANG)Orthogonal.o Orthogonal.cpp -c $(Flags)

Grassmann: Grassmann.cpp
	$(CXX) -o ../../obj/$(LANG)Grassmann.o Grassmann.cpp -c $(Flags)

TransRotInvPointCloud: TransRotInvPointCloud.cpp
	$(CXX) -o ../../obj/$(LANG)TransRotInvPointCloud.o TransRotInvPointCloud.cpp -c $(Flags) -DEIGEN_NO_STATIC_ASSERT
