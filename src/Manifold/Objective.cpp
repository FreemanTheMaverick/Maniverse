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

void Objective::Calculate(std::vector<EigenMatrix> /*P*/, int /*derivative*/){
	__Not_Implemented__
}

std::vector<EigenMatrix> Objective::Hessian(std::vector<EigenMatrix> X) const{
	__Not_Implemented__
	return std::vector<EigenMatrix>{X};
}

std::vector<EigenMatrix> Objective::Preconditioner(std::vector<EigenMatrix> X) const{
	return X;
}

std::vector<EigenMatrix> Objective::PreconditionerSqrt(std::vector<EigenMatrix> X) const{
	return X;
}

std::vector<EigenMatrix> Objective::PreconditionerInvSqrt(std::vector<EigenMatrix> X) const{
	return X;
}

#ifdef __PYTHON__
class PyObjective : public Objective, pybind11::trampoline_self_life_support{ public:
	using Objective::Objective;

	void Calculate(std::vector<EigenMatrix> P, int derivative) override{
		PYBIND11_OVERRIDE(void, Objective, Calculate, P, derivative);
	}

	std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> X) const override{
		PYBIND11_OVERRIDE(std::vector<EigenMatrix>, Objective, Hessian, X);
	}

	std::vector<EigenMatrix> Preconditioner(std::vector<EigenMatrix> X) const override{
		PYBIND11_OVERRIDE(std::vector<EigenMatrix>, Objective, Preconditioner, X);
	}

	std::vector<EigenMatrix> PreconditionerSqrt(std::vector<EigenMatrix> X) const override{
		PYBIND11_OVERRIDE(std::vector<EigenMatrix>, Objective, PreconditionerSqrt, X);
	}

	std::vector<EigenMatrix> PreconditionerInvSqrt(std::vector<EigenMatrix> X) const override{
		PYBIND11_OVERRIDE(std::vector<EigenMatrix>, Objective, PreconditionerInvSqrt, X);
	}
};

void Init_Objective(pybind11::module_& m){
	pybind11::classh<Objective, PyObjective>(m, "Objective")
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
