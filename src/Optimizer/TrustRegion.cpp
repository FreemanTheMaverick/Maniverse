#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <tuple>
#include <functional>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <string>
#include <tuple>
#include <memory>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "TrustRegion.h"

namespace Maniverse{

TrustRegion::TrustRegion(){
	this->R0 = 1;
	this->RhoThreshold = 0.1;
	this->Update = [&R0 = this->R0](double R, double Rho, double Snorm){
		if ( Rho < 0.25 ) R = std::min(0.25 * R, 0.75 * Snorm);
		else if ( Rho > 0.75 || std::abs(Snorm * Snorm - R * R) < 1.e-10 ) R = std::min(2 * R, R0);
		return R;
	};
}

#ifdef __PYTHON__
void Init_TrustRegion(pybind11::module_& m){
	pybind11::class_<TrustRegion>(m, "TrustRegion")
		.def_readwrite("R0", &TrustRegion::R0)
		.def_readwrite("RhoThreshold", &TrustRegion::RhoThreshold)
		.def_readwrite("Update", &TrustRegion::Update)
		.def(pybind11::init<>());
}
#endif

}
