#ifdef __PYTHON__

#include <pybind11/pybind11.h>

namespace Maniverse{

#define PyManiverseOut
#include "Manifold/PyManifold.h"
#include "Optimizer/PyOptimizer.h"
#undef PyManiverseOut

#define PyManiverseIn
PYBIND11_MODULE(Maniverse, m){
	#include "Manifold/PyManifold.h"
	#include "Optimizer/PyOptimizer.h"
}
#undef PyManiverseIn

}

#endif
