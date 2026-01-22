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
	for ( int iM = 0; iM < nMs; iM++ ) this->Ms.push_back(Ms[iM]);

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

	this->MatrixFree = matrix_free;
}

Iterate::Iterate(const Iterate& another_iterate){
	this->Func = another_iterate.Func;
	for ( auto& M : another_iterate.Ms ) this->Ms.push_back(M);
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

double Iterate::Inner(EigenVector X, EigenVector Y) const{
	double inner = 0;
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		inner += this->Ms[iM]->Inner(GetBlock(X, iM, this->BlockParameters), GetBlock(Y, iM, this->BlockParameters));
	}
	return inner;
}

EigenVector Iterate::Retract(EigenVector X) const{
	EigenVector Exp = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(Exp, iM, this->BlockParameters) = this->Ms[iM]->Retract(GetBlock(X, iM, this->BlockParameters));
	}
	return Exp;
}

EigenVector Iterate::InverseRetract(Iterate& N) const{
	EigenMatrix Log = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(Log, iM, this->BlockParameters) = this->Ms[iM]->InverseRetract(*(N.Ms[iM]));
	}
	return Log;
}

EigenVector Iterate::TransportTangent(EigenVector A, EigenVector Y) const{
	EigenVector B = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(B, iM, this->BlockParameters) = this->Ms[iM]->TransportTangent(GetBlock(A, iM, this->BlockParameters), GetBlock(Y, iM, this->BlockParameters));
	}
	return B;
}

EigenVector Iterate::TransportManifold(EigenVector A, Iterate& N) const{
	EigenVector B = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(B, iM, this->BlockParameters) = this->Ms[iM]->TransportManifold(GetBlock(A, iM, this->BlockParameters), *(N.Ms[iM]));
	}
	return B;
}

EigenVector Iterate::TangentProjection(EigenVector A) const{
	EigenVector X = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(X, iM, this->BlockParameters) = this->Ms[iM]->TangentProjection(GetBlock(A, iM, this->BlockParameters));
	}
	return X;
}

EigenVector Iterate::TangentPurification(EigenVector A) const{
	EigenVector X = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		SetBlock(X, iM, this->BlockParameters) = this->Ms[iM]->TangentPurification(GetBlock(A, iM, this->BlockParameters));
	}
	return X;
}

void Iterate::setPoint(std::vector<EigenMatrix> ps, bool purify){
	if ( ps.size() != this->Ms.size() ) throw std::runtime_error("Wrong number of Points!");
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		this->Ms[iM]->setPoint(ps[iM], purify);
		SetBlock(Point, iM, this->BlockParameters) = this->Ms[iM]->P;
	}
}

void Iterate::setGradient(){
	for ( int iM = 0; iM < (int)this->Ms.size(); iM++ ){
		this->Ms[iM]->Ge = this->Func->Gradient[iM];
		this->Ms[iM]->getGradient();
		SetBlock(Gradient, iM, this->BlockParameters) = this->Ms[iM]->Gr;
	}
}

std::vector<EigenMatrix> Iterate::getPoint() const{
	std::vector<EigenMatrix> ps(Ms.size());
	DecoupleBlock(this->Point, ps, this->BlockParameters);
	return ps;
}

std::vector<EigenMatrix> Iterate::getGradient() const{
	std::vector<EigenMatrix> gs;
	DecoupleBlock(this->Gradient, gs, this->BlockParameters);
	return gs;
}

EigenVector Iterate::Hessian(EigenVector Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<EigenMatrix> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM, this->BlockParameters);

	std::vector<EigenMatrix> HeX = this->Func->Hessian(X);

	EigenVector HrXmat = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(HrXmat, iM, this->BlockParameters) = this->Ms[iM]->getHessian(HeX[iM], X[iM], 1);
	}
	return HrXmat;
}

EigenVector Iterate::Preconditioner(EigenVector Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<EigenMatrix> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM, this->BlockParameters);

	std::vector<EigenMatrix> PX = this->Func->Preconditioner(X);

	EigenVector PXmat = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(PXmat, iM, this->BlockParameters) = PX[iM];
	}
	return PXmat;
}

EigenVector Iterate::PreconditionerSqrt(EigenVector Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<EigenMatrix> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM, this->BlockParameters);

	std::vector<EigenMatrix> PX = this->Func->PreconditionerSqrt(X);

	EigenVector PXmat = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(PXmat, iM, this->BlockParameters) = PX[iM];
	}
	return PXmat;
}

EigenVector Iterate::PreconditionerInvSqrt(EigenVector Xmat) const{
	const int nMs = (int)this->Ms.size();
	std::vector<EigenMatrix> X(nMs);
	for ( int iM = 0; iM < nMs; iM++ ) X[iM] = GetBlock(Xmat, iM, this->BlockParameters);

	std::vector<EigenMatrix> PX = this->Func->PreconditionerInvSqrt(X);

	EigenVector PXmat = EigenZero(this->TotalSize, 1);
	for ( int iM = 0; iM < nMs; iM++ ){
		SetBlock(PXmat, iM, this->BlockParameters) = PX[iM];
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
