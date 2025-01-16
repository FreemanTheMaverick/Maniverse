#include <pybind11/pybind11.h>
#include "Manifold/PyManifoldOut.h"

PYBIND11_MODULE(Maniverse, m){
	#include "Manifold/PyManifoldIn.h"
}
