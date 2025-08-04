#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <functional>
#include <tuple>
#include <memory>

#include "../Macro.h"

#include "Flag.h"
#include <iostream>

void Flag::setBlockParameters(std::vector<int> sizes){
	this->BlockParameters.clear();
	this->Name = "Flag(";
	int tot_size = 0;
	for ( int size : sizes ){
		if ( size <= 0 ) throw std::runtime_error("Invalid sizes of subspaces!");
		this->BlockParameters.push_back(std::make_tuple(tot_size, size));
		if ( tot_size > 0 ) this->Name += ", ";
		tot_size += size;
		this->Name += std::to_string(tot_size);
	}
	if ( tot_size > this->P.rows() ) throw std::runtime_error("Invalid sizes of subspaces!");
	this->Name += "; " + std::to_string(this->P.rows()) + ")";
}

Flag::Flag(EigenMatrix p): Stiefel(p){} // Be sure to Flag::setBlockParameters after construction.

int Flag::getDimension() const{
	const int N = this->P.rows();
	int ndim = 0;
	int n = 0;
	for ( auto block_parameter : this->BlockParameters ){
		const int delta_n = std::get<1>(block_parameter);
		n += delta_n;
		ndim += delta_n * ( N - n );
	}
	return ndim;
}

inline static EigenMatrix TangentProjection(EigenMatrix P, std::vector<std::tuple<int, int>> BlockParameters, EigenMatrix X){
	EigenMatrix Y = X;
	for ( int i = 0; i < (int)BlockParameters.size(); i++ ){
		FlagGetBlock(Y, i) -= FlagGetBlock(P, i) * FlagGetBlock(P, i).transpose() * FlagGetBlock(X, i);
		for ( int j = 0; j < (int)BlockParameters.size(); j++ ){
			if ( i != j ) FlagGetBlock(Y, i) -= FlagGetBlock(P, j) * FlagGetBlock(X, j).transpose() * FlagGetBlock(P, i);
		}
	}
	return Y;
}

EigenMatrix Flag::TangentProjection(EigenMatrix X) const{
	return ::TangentProjection(this->P, this->BlockParameters, X);
}

std::function<EigenMatrix (EigenMatrix)> Flag::getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool weingarten) const{
	//https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/stiefel
	const EigenMatrix P = this->P;
	const std::vector<std::tuple<int, int>> B = this->BlockParameters;
	const EigenMatrix tmp = this->Ge.transpose() * this->P + this->P.transpose() * this->Ge;
	if ( weingarten ) return [P, B, tmp, He](EigenMatrix v){
		return ::TangentProjection(P, B, He(v) - 0.5 * v * tmp);
	};
	else return [P, B, He](EigenMatrix v){
		return ::TangentProjection(P, B, He(v));
	};
}

std::unique_ptr<Manifold> Flag::Clone() const{
	return std::make_unique<Flag>(*this);
}

#ifdef __PYTHON__
void Init_Flag(pybind11::module_& m){
	pybind11::classh<Flag, Stiefel>(m, "Flag")
		.def(pybind11::init<EigenMatrix>())
		.def("setBlockParameters", &Flag::setBlockParameters);
}
#endif
