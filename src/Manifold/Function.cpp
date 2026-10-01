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

void Function::Calculate(std::vector<Eigen::MatrixXd> /*P*/, std::vector<int> /*derivatives*/){
	__Not_Implemented__
}

std::vector<Eigen::MatrixXd> Function::Hessian(std::vector<Eigen::MatrixXd> X) const{
	__Not_Implemented__
	return std::vector<Eigen::MatrixXd>{X};
}

std::vector<Eigen::MatrixXd> Function::Preconditioner(std::vector<Eigen::MatrixXd> X) const{
	return X;
}

std::vector<Eigen::MatrixXd> Function::PreconditionerInv(std::vector<Eigen::MatrixXd> X) const{
	return X;
}

std::vector<Eigen::MatrixXd> Function::PreconditionerSqrt(std::vector<Eigen::MatrixXd> X) const{
	return X;
}

std::vector<Eigen::MatrixXd> Function::PreconditionerInvSqrt(std::vector<Eigen::MatrixXd> X) const{
	return X;
}

#ifdef __PYTHON__
class PyFunction : public Function, pybind11::trampoline_self_life_support{ public:
	using Function::Function;

	void Calculate(std::vector<Eigen::MatrixXd> P, std::vector<int> derivatives) override{
		PYBIND11_OVERRIDE(void, Function, Calculate, P, derivatives);
	}

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> X) const override{
		PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, Function, Hessian, X);
	}

	std::vector<Eigen::MatrixXd> Preconditioner(std::vector<Eigen::MatrixXd> X) const override{
		PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, Function, Preconditioner, X);
	}

	std::vector<Eigen::MatrixXd> PreconditionerSqrt(std::vector<Eigen::MatrixXd> X) const override{
		PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, Function, PreconditionerSqrt, X);
	}

	std::vector<Eigen::MatrixXd> PreconditionerInvSqrt(std::vector<Eigen::MatrixXd> X) const override{
		PYBIND11_OVERRIDE(std::vector<Eigen::MatrixXd>, Function, PreconditionerInvSqrt, X);
	}
};

void Init_Function(pybind11::module_& m){
	pybind11::classh<Function, PyFunction>(m, "Function")
		.def(pybind11::init<>())
		.def("Calculate", &Function::Calculate)
		.def_readwrite("Value", &Function::Value)
		.def_readwrite("Gradient", &Function::Gradient)
		.def("Hessian", &Function::Hessian)
		.def("Preconditioner", &Function::Preconditioner)
		.def("PreconditionerSqrt", &Function::PreconditionerSqrt)
		.def("PreconditionerInvSqrt", &Function::PreconditionerInvSqrt);
}
#endif

}
