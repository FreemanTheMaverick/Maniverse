#ifdef __PYTHON__
#include <pybind11/pybind11.h>

#include "AugmentedLagrangian.h"

namespace Maniverse{

void Init_AugmentedLagrangian(pybind11::module_& m){
	m.def("AugmentedLagrangian", &AugmentedLagrangian);
}

}
#endif
