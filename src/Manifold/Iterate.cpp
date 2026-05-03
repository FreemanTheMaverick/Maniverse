#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <tuple>

#include "Manifold.h"

namespace Maniverse{

Iterate::Iterate(Objective& func, std::vector<std::shared_ptr<Manifold>> Ms){
	this->Func = &func;

	const int nMs = (int)Ms.size();
	this->Ms = Ms;

	this->TotalSize = 0;
	for ( int iM = 0; iM < nMs; iM++ ){
		this->BlockParameters.push_back(std::make_tuple(
				this->TotalSize,
				this->Ms[iM]->P.rows(),
				this->Ms[iM]->P.cols()
		));
		this->TotalSize += this->Ms[iM]->P.size();
	}

	this->Point.resize(this->TotalSize); this->Point.setZero();
	this->Gradient.resize(this->TotalSize); this->Gradient.setZero();
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(Point, iM, this->BlockParameters) = Ms[iM]->P;
		SetBlock(Gradient, iM, this->BlockParameters) = Ms[iM]->Gr;
	}

	for ( int icons = 0; icons < (int)this->Func->Lambda.size(); icons++ ){
		this->Constraints.push_back({});
		this->Constraint_Gradient.push_back(Eigen::VectorXd::Zero(this->TotalSize));
		for ( int jM = 0; jM < nMs; jM++ ){
			this->Constraints[icons].push_back(Ms[jM]->Share());
		}
	}
}

std::string Iterate::getName() const{
	std::string name = "";
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		if ( iM > 0 ) name += " * ";
		name += Ms[iM]->Name;
	}
	return name;
}

int Iterate::getDimension() const{
	int ndims = 0;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ )
		 ndims += Ms[iM]->getDimension();
	return ndims;
}

double Iterate::Inner(Eigen::VectorXd X, Eigen::VectorXd Y) const{
	double inner = 0;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		inner += this->Ms[iM]->Inner(GetBlock(X, iM, this->BlockParameters), GetBlock(Y, iM, this->BlockParameters));
	}
	return inner;
}

Eigen::VectorXd Iterate::Retract(Eigen::VectorXd X) const{
	Eigen::VectorXd Exp = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(Exp, iM, this->BlockParameters) = this->Ms[iM]->Retract(GetBlock(X, iM, this->BlockParameters));
	}
	return Exp;
}

Eigen::VectorXd Iterate::InverseRetract(Iterate& N) const{
	Eigen::MatrixXd Log = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(Log, iM, this->BlockParameters) = this->Ms[iM]->InverseRetract(*(N.Ms[iM]));
	}
	return Log;
}

Eigen::VectorXd Iterate::TransportTangent(Eigen::VectorXd A, Eigen::VectorXd Y) const{
	Eigen::VectorXd B = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(B, iM, this->BlockParameters) = this->Ms[iM]->TransportTangent(GetBlock(A, iM, this->BlockParameters), GetBlock(Y, iM, this->BlockParameters));
	}
	return B;
}

Eigen::VectorXd Iterate::TransportManifold(Eigen::VectorXd A, Iterate& N) const{
	Eigen::VectorXd B = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(B, iM, this->BlockParameters) = this->Ms[iM]->TransportManifold(GetBlock(A, iM, this->BlockParameters), *(N.Ms[iM]));
	}
	return B;
}

Eigen::VectorXd Iterate::TangentProjection(Eigen::VectorXd A) const{
	Eigen::VectorXd X = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(X, iM, this->BlockParameters) = this->Ms[iM]->TangentProjection(GetBlock(A, iM, this->BlockParameters));
	}
	return X;
}

Eigen::VectorXd Iterate::TangentPurification(Eigen::VectorXd A) const{
	Eigen::VectorXd X = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(X, iM, this->BlockParameters) = this->Ms[iM]->TangentPurification(GetBlock(A, iM, this->BlockParameters));
	}
	return X;
}

void Iterate::setPoint(std::vector<Eigen::MatrixXd> ps, bool purify){
	if ( ps.size() != this->Ms.size() ) throw std::runtime_error("Wrong number of Points!");
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		this->Ms[iM]->setPoint(ps[iM], purify);
		SetBlock(Point, iM, this->BlockParameters) = this->Ms[iM]->P;
	}
	for ( int icons = 0; icons < (int)this->Constraints.size(); icons++ ){
		for ( int jM = 0; jM < (int)this->Ms.size(); jM++ ){
			this->Constraints[icons][jM]->setPoint(ps[jM], purify);
		}
	}
}

void Iterate::setGradient(){
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		this->Ms[iM]->Ge = this->Func->Gradient[iM];
		this->Ms[iM]->getGradient();
		SetBlock(Gradient, iM, this->BlockParameters) = this->Ms[iM]->Gr;
	}
	for ( int icons = 0; icons < (int)this->Constraints.size(); icons++ ){
		for ( int jM = 0; jM < (int)this->Ms.size(); jM++ ){
			this->Constraints[icons][jM]->Ge = this->Func->Constraint_Gradient[icons][jM];
			this->Constraints[icons][jM]->getGradient();
			Eigen::VectorXd& cons_grad_i = this->Constraint_Gradient[icons];
			SetBlock(cons_grad_i, jM, this->BlockParameters) = this->Constraints[icons][jM]->Gr;
		}
	}
}

std::vector<Eigen::MatrixXd> Iterate::getPoint() const{
	std::vector<Eigen::MatrixXd> ps(Ms.size());
	DecoupleBlock(this->Point, ps, this->BlockParameters);
	return ps;
}

std::vector<Eigen::MatrixXd> Iterate::getGradient() const{
	std::vector<Eigen::MatrixXd> gs;
	DecoupleBlock(this->Gradient, gs, this->BlockParameters);
	return gs;
}

Eigen::VectorXd Iterate::Hessian(Eigen::VectorXd Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<Eigen::MatrixXd> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM, this->BlockParameters);

	std::vector<Eigen::MatrixXd> HeX = this->Func->Hessian(X);

	Eigen::VectorXd HrXmat = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(HrXmat, iM, this->BlockParameters) = this->Ms[iM]->getHessian(HeX[iM], X[iM], 1);
	}
	return HrXmat;
}

std::vector<double> Iterate::getEffectiveLambda() const{
	const int ncons = this->Func->Lambda.size();
	Eigen::VectorXd Gf = this->Gradient;
	Eigen::MatrixXd Gg = Eigen::MatrixXd::Zero(Gf.size(), ncons);
	for ( int i = 0; i < ncons; i++ ){
		Gf -= this->Func->Lambda[i] * this->Constraint_Gradient[i];
		Gg.col(i) = this->Constraint_Gradient[i];
	}
	const Eigen::VectorXd lambda = - Gg.colPivHouseholderQr().solve(Gf);
	return std::vector<double>(lambda.data(), lambda.data() + ncons);
}

Eigen::VectorXd Iterate::ConstraintProjection(Eigen::VectorXd Xmat) const{
	for ( const Eigen::VectorXd& cons_grad : this->Constraint_Gradient ){
		Xmat -= this->Inner(Xmat, cons_grad) * cons_grad / this->Inner(cons_grad, cons_grad);
	}
	return Xmat;
}

Eigen::VectorXd Iterate::ConstraintProjectedHessian(Eigen::VectorXd Xmat) const{
	// Xmat must observe the constraints.
	const double Rho = this->Func->Rho;
	this->Func->Rho = 0;
	const Eigen::VectorXd HXmat = this->ConstraintProjection(this->Hessian(Xmat));
	this->Func->Rho = Rho;
	return HXmat;
}

Eigen::VectorXd Iterate::Preconditioner(Eigen::VectorXd Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<Eigen::MatrixXd> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM, this->BlockParameters);

	std::vector<Eigen::MatrixXd> PX = this->Func->Preconditioner(X);

	Eigen::VectorXd PXmat = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(PXmat, iM, this->BlockParameters) = PX[iM];
	}
	return PXmat;
}

Eigen::VectorXd Iterate::ConstraintProjectedPreconditioner(Eigen::VectorXd Xmat) const{
	// Xmat must observe the constraints.
	const double Rho = this->Func->Rho;
	this->Func->Rho = 0;
	const Eigen::VectorXd PXmat = this->ConstraintProjection(this->Preconditioner(Xmat));
	this->Func->Rho = Rho;
	return PXmat;
}

Eigen::VectorXd Iterate::PreconditionerInv(Eigen::VectorXd Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<Eigen::MatrixXd> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM, this->BlockParameters);

	std::vector<Eigen::MatrixXd> PX = this->Func->PreconditionerInv(X);

	Eigen::VectorXd PXmat = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(PXmat, iM, this->BlockParameters) = PX[iM];
	}
	return PXmat;
}

Eigen::VectorXd Iterate::ConstraintProjectedPreconditionerInv(Eigen::VectorXd Xmat) const{
	// Xmat must observe the constraints.
	const double Rho = this->Func->Rho;
	this->Func->Rho = 0;
	const Eigen::VectorXd PXmat = this->ConstraintProjection(this->PreconditionerInv(Xmat));
	this->Func->Rho = Rho;
	return PXmat;
}

Eigen::VectorXd Iterate::PreconditionerSqrt(Eigen::VectorXd Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<Eigen::MatrixXd> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM, this->BlockParameters);

	std::vector<Eigen::MatrixXd> PX = this->Func->PreconditionerSqrt(X);

	Eigen::VectorXd PXmat = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(PXmat, iM, this->BlockParameters) = PX[iM];
	}
	return PXmat;
}

Eigen::VectorXd Iterate::PreconditionerInvSqrt(Eigen::VectorXd Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<Eigen::MatrixXd> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM, this->BlockParameters);

	std::vector<Eigen::MatrixXd> PX = this->Func->PreconditionerInvSqrt(X);

	Eigen::VectorXd PXmat = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(PXmat, iM, this->BlockParameters) = PX[iM];
	}
	return PXmat;
}

#ifdef __PYTHON__
void Init_Iterate(pybind11::module_& m){
	pybind11::classh<Iterate>(m, "Iterate")
		.def_readwrite("Ms", &Iterate::Ms)
		.def_readwrite("Func", &Iterate::Func)
		.def_readwrite("Point", &Iterate::Point)
		.def_readwrite("Gradient", &Iterate::Gradient)
		.def("Hessian", &Iterate::Hessian)
		.def("ConstraintProjectedHessian", &Iterate::ConstraintProjectedHessian)
		.def("Preconditioner", &Iterate::Preconditioner)
		.def("ConstraintProjectedPreconditioner", &Iterate::ConstraintProjectedPreconditioner)
		.def("PreconditionerSqrt", &Iterate::PreconditionerSqrt)
		.def("PreconditionerInvSqrt", &Iterate::PreconditionerInvSqrt)
		.def_readwrite("Constraints", &Iterate::Constraints)
		.def_readwrite("Constraint_Gradient", &Iterate::Constraint_Gradient)
		.def_readwrite("TotalSize", &Iterate::TotalSize)
		.def_readwrite("BlockParameters", &Iterate::BlockParameters)
		.def(pybind11::init<Objective&, std::vector<std::shared_ptr<Manifold>>>())
		.def("getName", &Iterate::getName)
		.def("getDimension", &Iterate::getDimension)
		.def("Inner", &Iterate::Inner)
		.def("Retract", &Iterate::Retract)
		.def("InverseRetract", &Iterate::InverseRetract)
		.def("TangentProjection", &Iterate::TangentProjection)
		.def("TangentPurification", &Iterate::TangentPurification)
		.def("ConstraintProjection", &Iterate::ConstraintProjection)
		.def("TransportManifold", &Iterate::TransportManifold)
		.def("setPoint", &Iterate::setPoint)
		.def("setGradient", &Iterate::setGradient)
		.def("getPoint", &Iterate::getPoint)
		.def("getGradient", &Iterate::getGradient)
		.def("getEffectiveLambda", &Iterate::getEffectiveLambda);
}
#endif

}
