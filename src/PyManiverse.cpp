#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include "Manifold/PyManifoldOut.h"
#include "Optimizer/PyOptimizerOut.h"

PYBIND11_MODULE(Maniverse, m){
	#include "Manifold/PyManifoldIn.h"
	#include "Optimizer/PyOptimizerIn.h"
}
#endif
