#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <vector>

#include "../Macro.h"

#include "Manifold.h"

namespace Maniverse{

void Objective::Calculate(std::vector<Eigen::MatrixXd> /*P*/, std::vector<int> /*derivatives*/){
	__Not_Implemented__
}

std::vector<Eigen::MatrixXd> Objective::Hessian(std::vector<Eigen::MatrixXd> X) const{
	__Not_Implemented__
	return std::vector<Eigen::MatrixXd>{X};
}

std::vector<Eigen::MatrixXd> Objective::Preconditioner(std::vector<Eigen::MatrixXd> X) const{
	return X;
}

std::vector<Eigen::MatrixXd> Objective::PreconditionerSqrt(std::vector<Eigen::MatrixXd> X) const{
	return X;
}

std::vector<Eigen::MatrixXd> Objective::PreconditionerInvSqrt(std::vector<Eigen::MatrixXd> X) const{
	return X;
}

#ifdef __PYTHON__
class PyObjective : public Objective, pybind11::trampoline_self_life_support{ public:
	using Objective::Objective;

	void Calculate(std::vector<Eigen::MatrixXd> P, std::vector<int> derivatives) override{
		PYBIND11_OVERRIDE(void, Objective, Calculate, P, derivatives);
	}

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> X) const override{
		PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, Objective, Hessian, X);
	}

	std::vector<Eigen::MatrixXd> Preconditioner(std::vector<Eigen::MatrixXd> X) const override{
		PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, Objective, Preconditioner, X);
	}

	std::vector<Eigen::MatrixXd> PreconditionerSqrt(std::vector<Eigen::MatrixXd> X) const override{
		PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, Objective, PreconditionerSqrt, X);
	}

	std::vector<Eigen::MatrixXd> PreconditionerInvSqrt(std::vector<Eigen::MatrixXd> X) const override{
		PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, Objective, PreconditionerInvSqrt, X);
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
		.def("PreconditionerInvSqrt", &Objective::PreconditionerInvSqrt)
		.def_readwrite("Lambda", &Objective::Lambda)
		.def_readwrite("Rho", &Objective::Rho)
		.def_readwrite("Constraint_Value", &Objective::Constraint_Value)
		.def_readwrite("Constraint_Gradient", &Objective::Constraint_Gradient);
}
#endif

}
