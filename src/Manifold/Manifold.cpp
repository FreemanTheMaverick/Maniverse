#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Core>

#include "../Macro.h"
#include "Manifold.h"

void Init_Manifold(pybind11::module_& m){
	pybind11::class_<Manifold>(m, "Manifold")
		.def_readwrite("Name", &Manifold::Name)
		.def_readwrite("P", &Manifold::P)
		.def_readwrite("Aux", &Manifold::Aux)
		.def_readwrite("Ge", &Manifold::Ge)
		.def_readwrite("Gr", &Manifold::Gr)
		.def_readwrite("He", &Manifold::He)
		.def_readwrite("Hr", &Manifold::Hr)
		.def("getDimension", &Manifold::getDimension)
		.def("Inner", &Manifold::Inner)
		.def("getInner", &Manifold::getInner)
		.def("Distance", &Manifold::Distance)
		.def("Exponential", &Manifold::Exponential)
		.def("Logarithm", &Manifold::Logarithm)
		.def("TangentProjection", &Manifold::TangentProjection)
		.def("TangentPurification", &Manifold::TangentPurification)
		.def("TransportTangent", &Manifold::TransportTangent)
		.def("TransportManifold", &Manifold::TransportManifold)
		.def("Update", &Manifold::Update)
		.def("getGradient", &Manifold::getGradient)
		.def("getHessian", &Manifold::getHessian);
}
