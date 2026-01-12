#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif
#include <Eigen/Dense>
#include <typeinfo>
#include <memory>

#include "../Macro.h"

#include "Manifold.h"

namespace Maniverse{

Iterate::Iterate(Objective& func, std::vector<std::shared_ptr<Manifold>> Ms, bool matrix_free){
	this->Func = &func;

	const int nMs = (int)Ms.size();
	this->Ms.clear();
	for ( int iM = 0; iM < nMs; iM++ ) this->Ms.push_back(Ms[iM]->Clone());

	int nrows = 0;
	int ncols = 0;
	for ( int iM = 0; iM < nMs; iM++ ){
		this->BlockParameters.push_back(std::make_tuple(
				nrows, ncols,
				this->Ms[iM]->P.rows(),
				this->Ms[iM]->P.cols()
		));
		nrows += this->Ms[iM]->P.rows();
		ncols += this->Ms[iM]->P.cols();
	}

	this->Point.resize(nrows, ncols); this->Point.setZero();
	this->Gradient.resize(nrows, ncols); this->Gradient.setZero();
	for ( int iM = 0; iM < (int)Ms.size(); iM++ ){
		GetBlock(this->Point, iM) = Ms[iM]->P;
		GetBlock(this->Gradient, iM) = Ms[iM]->Gr;
	}

	this->MatrixFree = matrix_free;
}

Iterate::Iterate(const Iterate& another_iterate){
	this->Func = another_iterate.Func;
	for ( auto& M : another_iterate.Ms ) this->Ms.push_back(M->Clone());
	this->Point = another_iterate.Point;
	this->Gradient = another_iterate.Gradient;
	this->MatrixFree = another_iterate.MatrixFree;
	this->BasisSet = another_iterate.BasisSet;
	this->HessianMatrix = another_iterate.HessianMatrix;
	this->BlockParameters = another_iterate.BlockParameters;
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

double Iterate::Inner(EigenMatrix X, EigenMatrix Y) const{
	double inner = 0;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		inner += this->Ms[iM]->Inner(GetBlock(X, iM), GetBlock(Y, iM));
	}
	return inner;
}

EigenMatrix Iterate::Retract(EigenMatrix X) const{
	EigenMatrix Exp = EigenZero(X.rows(), X.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(Exp, iM) = this->Ms[iM]->Retract(GetBlock(X, iM));
	}
	return Exp;
}

EigenMatrix Iterate::InverseRetract(Iterate& N) const{
	EigenMatrix Log = EigenZero(this->Point.rows(), this->Point.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(Log, iM) = this->Ms[iM]->InverseRetract(*(N.Ms[iM]));
	}
	return Log;
}

EigenMatrix Iterate::TransportTangent(EigenMatrix A, EigenMatrix Y) const{
	EigenMatrix B = EigenZero(A.rows(), A.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(B, iM) = this->Ms[iM]->TransportTangent(GetBlock(A, iM), GetBlock(Y, iM));
	}
	return B;
}

EigenMatrix Iterate::TransportManifold(EigenMatrix A, Iterate& N) const{
	EigenMatrix B = EigenZero(A.rows(), A.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(B, iM) = this->Ms[iM]->TransportManifold(GetBlock(A, iM), *(N.Ms[iM]));
	}
	return B;
}

EigenMatrix Iterate::TangentProjection(EigenMatrix A) const{
	EigenMatrix X = EigenZero(A.rows(), A.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(X, iM) = this->Ms[iM]->TangentProjection(GetBlock(A, iM));
	}
	return X;
}

EigenMatrix Iterate::TangentPurification(EigenMatrix A) const{
	EigenMatrix X = EigenZero(A.rows(), A.cols());
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		GetBlock(X, iM) = this->Ms[iM]->TangentPurification(GetBlock(A, iM));
	}
	return X;
}

void Iterate::setPoint(std::vector<EigenMatrix> ps, bool purify){
	if ( ps.size() != this->Ms.size() ) throw std::runtime_error("Wrong number of Points!");
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		this->Ms[iM]->setPoint(ps[iM], purify);
		GetBlock(this->Point, iM) = this->Ms[iM]->P;
	}
}

void Iterate::setGradient(){
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		this->Ms[iM]->Ge = this->Func->Gradient[iM];
		this->Ms[iM]->getGradient();
		GetBlock(this->Gradient, iM) = this->Ms[iM]->Gr;
	}
}

std::vector<EigenMatrix> Iterate::getPoint() const{
	std::vector<EigenMatrix> ps;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		ps.push_back(GetBlock(this->Point, iM));
	}
	return ps;
}

std::vector<EigenMatrix> Iterate::getGradient() const{
	std::vector<EigenMatrix> gs;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		gs.push_back(GetBlock(this->Gradient, iM));
	}
	return gs;
}

EigenMatrix Iterate::Hessian(EigenMatrix Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<EigenMatrix> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM);

	std::vector<std::vector<EigenMatrix>> HeX = this->Func->Hessian(X);

	EigenMatrix HrXmat = EigenZero(Xmat.rows(), Xmat.cols());
	for ( int iM = 0, khess = 0; iM < nMs; iM++ ) for ( int jM = 0; jM < nMs; jM++, khess++ ){
		GetBlock(HrXmat, iM) += this->Ms[iM]->getHessian(HeX[iM][jM], X[jM], iM == jM);
	}
	return HrXmat;
}

EigenMatrix Iterate::Preconditioner(EigenMatrix Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<EigenMatrix> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM);

	std::vector<std::vector<EigenMatrix>> PX = this->Func->Preconditioner(X);

	EigenMatrix PXmat = EigenZero(Xmat.rows(), Xmat.cols());
	for ( int iM = 0, khess = 0; iM < nMs; iM++ ) for ( int jM = 0; jM < nMs; jM++, khess++ ){
		GetBlock(PXmat, iM) += PX[iM][jM];
	}
	return PXmat;
}

EigenMatrix Iterate::PreconditionerSqrt(EigenMatrix Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<EigenMatrix> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM);

	std::vector<std::vector<EigenMatrix>> PX = this->Func->PreconditionerSqrt(X);

	EigenMatrix PXmat = EigenZero(Xmat.rows(), Xmat.cols());
	for ( int iM = 0, khess = 0; iM < nMs; iM++ ) for ( int jM = 0; jM < nMs; jM++, khess++ ){
		GetBlock(PXmat, iM) += PX[iM][jM];
	}
	return PXmat;
}

EigenMatrix Iterate::PreconditionerInvSqrt(EigenMatrix Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<EigenMatrix> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM);

	std::vector<std::vector<EigenMatrix>> PX = this->Func->PreconditionerInvSqrt(X);

	EigenMatrix PXmat = EigenZero(Xmat.rows(), Xmat.cols());
	for ( int iM = 0, khess = 0; iM < nMs; iM++ ) for ( int jM = 0; jM < nMs; jM++, khess++ ){
		GetBlock(PXmat, iM) += PX[iM][jM];
	}
	return PXmat;
}

#ifdef __PYTHON__
void Init_Iterate(pybind11::module_& m){
	pybind11::classh<Iterate>(m, "Iterate")
		.def_readonly("Ms", &Iterate::Ms)
		.def_readwrite("Point", &Iterate::Point)
		.def_readwrite("Gradient", &Iterate::Gradient)
		.def("Hessian", &Iterate::Hessian)
		.def("Preconditioner", &Iterate::Preconditioner)
		.def("PreconditionerSqrt", &Iterate::PreconditionerSqrt)
		.def("PreconditionerInvSqrt", &Iterate::PreconditionerInvSqrt)
		.def_readwrite("MatrixFree", &Iterate::MatrixFree)
		.def_readwrite("BlockParameters", &Iterate::BlockParameters)
		.def(pybind11::init<Objective&, std::vector<std::shared_ptr<Manifold>>, bool>())
		.def(pybind11::init<const Iterate&>())
		.def("getName", &Iterate::getName)
		.def("getDimension", &Iterate::getDimension)
		.def("Inner", &Iterate::Inner)
		.def("Retract", &Iterate::Retract)
		.def("InverseRetract", &Iterate::InverseRetract)
		.def("TangentProjection", &Iterate::TangentProjection)
		.def("TangentPurification", &Iterate::TangentPurification)
		.def("TransportManifold", &Iterate::TransportManifold)
		.def("setPoint", &Iterate::setPoint)
		.def("setGradient", &Iterate::setGradient);
}
#endif

}
