#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif
#include <Eigen/Dense>
#include <typeinfo>
#include <memory>
#include <vector>

#include "../Macro.h"

#include "Manifold.h"

namespace Maniverse{

void Objective::Calculate(std::vector<EigenMatrix>& /*P*/){
	__Not_Implemented__
}

std::vector<EigenMatrix> Objective::Hessian(std::vector<EigenMatrix>& X){
	__Not_Implemented__
	return X;
}

std::vector<EigenMatrix> Objective::Preconditioner(std::vector<EigenMatrix>& X){
	__Not_Implemented__
	return X;
}

std::vector<EigenMatrix> Objective::PreconditionerSqrt(std::vector<EigenMatrix>& X){
	__Not_Implemented__
	return X;
}

std::vector<EigenMatrix> Objective::PreconditionerInvSqrt(std::vector<EigenMatrix>& X){
	__Not_Implemented__
	return X;
}

#ifdef __PYTHON__
void Init_Objective(pybind11::module_& m){
	pybind11::classh<Objective>(m, "Objective")
		.def(pybind11::init<>())
		.def("Calculate", &Objective::Calculate)
		.def_readwrite("Value", &Objective::Value)
		.def_readwrite("Gradient", &Objective::Gradient)
		.def("Hessian", &Objective::Hessian)
		.def("Preconditioner", &Objective::Preconditioner)
		.def("PreconditionerSqrt", &Objective::PreconditionerSqrt)
		.def("PreconditionerInvSqrt", &Objective::PreconditionerInvSqrt);
}
#endif

}
