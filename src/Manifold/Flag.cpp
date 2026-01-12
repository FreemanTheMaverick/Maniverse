#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>
#include <tuple>
#include <memory>

#include "../Macro.h"

#include "Flag.h"

namespace Maniverse{

void Flag::setBlockParameters(std::vector<int> sizes){
	this->BlockParameters.clear();
	this->Name = "Flag(";
	int tot_size = 0;
	for ( int size : sizes ){
		if ( size <= 0 ) throw std::runtime_error("Invalid sizes of subspaces! (Non-positive dimension)");
		this->BlockParameters.push_back(std::make_tuple(tot_size, size));
		if ( tot_size > 0 ) this->Name += ", ";
		tot_size += size;
		this->Name += std::to_string(tot_size);
	}
	if ( tot_size > this->P.cols() ) throw std::runtime_error("Invalid sizes of subspaces! (Dimension exceeds)");
	this->Name += "; " + std::to_string(this->P.rows()) + ")";
}

Flag::Flag(EigenMatrix p, std::string geodesic): Stiefel(p, geodesic){} // Be sure to Flag::setBlockParameters after construction.

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

static EigenMatrix symf(std::vector<std::tuple<int, int>> BlockParameters, EigenMatrix A){
	// https://doi.org/10.1007/s10957-023-02242-z
	for ( int i = 0; i < (int)BlockParameters.size(); i++ ){
		for ( int j = 0; j < i; j++ ){
			FlagGetBlock(A, i, j) = ( FlagGetBlock(A, i, j) + FlagGetBlock(A, j, i).transpose() ) / 2;
			FlagGetBlock(A, j, i) = FlagGetBlock(A, i, j).transpose();
		}
	}
	return A;
}

static EigenMatrix FlagTangentProjection(EigenMatrix P, std::vector<std::tuple<int, int>> BlockParameters, EigenMatrix X){
	// https://doi.org/10.1007/s10957-023-02242-z
	return X - P * symf(BlockParameters, P.transpose() * X);
}

EigenMatrix Flag::TangentProjection(EigenMatrix X) const{
	return FlagTangentProjection(this->P, this->BlockParameters, X);
}

EigenMatrix Flag::TangentPurification(EigenMatrix X) const{
	return FlagTangentProjection(this->P, this->BlockParameters, X);
}

EigenMatrix Flag::getHessian(EigenMatrix HeX, EigenMatrix X, bool weingarten) const{
	// https://doi.org/10.1007/s10957-023-02242-z
	if ( weingarten ){
		const EigenMatrix tmp = symf(this->BlockParameters, this->P.transpose() * this->Ge);
		return this->TangentProjection(HeX - X * tmp);
	}else return this->TangentProjection(HeX);
}

std::unique_ptr<Manifold> Flag::Clone() const{
	return std::make_unique<Flag>(*this);
}

#ifdef __PYTHON__
void Init_Flag(pybind11::module_& m){
	pybind11::classh<Flag, Stiefel>(m, "Flag")
		.def(pybind11::init<EigenMatrix, std::string>(), pybind11::arg("p"), pybind11::arg("geodesic") = "POLAR")
		.def("setBlockParameters", &Flag::setBlockParameters);
}
#endif

}
