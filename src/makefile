.PHONY: all

all: MANIFOLD PYMANIVERSE

MANIFOLD:
	$(MAKE) -C Manifold


PYMANIVERSE: PyManiverse.cpp
	$(CXX) -o ../obj/PyManiverse.o PyManiverse.cpp -c $(Flags) $(GeneralFlags) $(PYBIND11Flags)
