.PHONY: all

all: MANIFOLD OPTIMIZER PYMANIVERSE

MANIFOLD:
	$(MAKE) -C Manifold

OPTIMIZER:
	$(MAKE) -C Optimizer

PYMANIVERSE: PyManiverse.cpp
	$(CXX) -o ../obj/$(OBJ)PyManiverse.o PyManiverse.cpp -c $(Flags)
