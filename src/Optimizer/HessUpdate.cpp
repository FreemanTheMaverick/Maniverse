#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif
#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <vector>
#include <map>
#include <functional>
#include <string>
#include <cassert>
#include <memory>
#include <cstdio>

#include "../Macro.h"
#include "../Manifold/Manifold.h"
#include "HessUpdate.h"

#include <iostream>


HessUpdate::HessUpdate(int n){
	this->Size = n;
	this->Ms.reserve(n);
	this->Caches.reserve(n);
}

void HessUpdate::Append(Manifold& M, EigenMatrix Step){
	if ( (int)this->Ms.size() >= this->Size || (int)this->Caches.size() >= this->Size ){
		throw std::runtime_error("The number of previous steps exceeds the prescribed size!");
	}
	this->AdmittedAppend(M, Step);
}

void HessUpdate::AdmittedAppend(Manifold& M, EigenMatrix Step){
	__Not_Implemented__
	M.P += Step - Step;
	this->Ms.push_back(M.Clone());
}

EigenMatrix HessUpdate::Hessian(EigenMatrix v){
	if (this->Ms[0]->MatrixFree) return this->HessianMatrixFree(v);
	else{
		EigenMatrix Hv = EigenZero(v.rows(), v.cols());
		for ( auto& [eigenvalue, eigenvector] : this->EigenPairs ){
			Hv += eigenvalue * eigenvector.cwiseProduct(v).sum() * eigenvector;
		}
		return Hv;
	}
}

EigenMatrix HessUpdate::HessianMatrixFree(EigenMatrix v){
	__Not_Implemented__
	return v;
}

void HessUpdate::Clear(){
	this->Ms.clear();
	this->Caches.clear();
	this->EigenPairs.clear();
}

void BroydenFletcherGoldfarbShanno::AdmittedAppend(Manifold& M, EigenMatrix step){
	if (M.MatrixFree){
		std::map<std::string, EigenMatrix> cache;
		if ( ! this->Ms.empty() ){
			const EigenMatrix S = cache["S"] = this->Ms.back()->TransportManifold(step, M);
			const EigenMatrix Y = cache["Y"] = M.Gr - this->Ms.back()->TransportManifold(this->Ms.back()->Gr, M);
			const double YS = M.Inner(Y, S);
			if ( this->Verbose ){
				const double score = YS / M.Inner(S, S);
				const double threshold = this->CautiousThreshold(std::sqrt(this->Ms.back()->Inner(this->Ms.back()->Gr, this->Ms.back()->Gr)));
				const bool update = score > threshold;
				std::printf("Score of hessian update:\n");
				std::printf("| Score ( <Y, S> / <S, S> ): %E\n", score);
				std::printf("| Threshold ( CautiousThresholdFunc(sqrt(<Gr, Gr>)) ): %E\n", threshold);
				std::printf("| Update: %s\n", __True_False__(update));
			}
			cache["YoverYS"] = Y / YS;
			const EigenMatrix HS = this->Ms.back()->TransportManifold(this->Hessian(step), M);
			cache["HSoverSHS"] = HS / M.Inner(S, HS);
		}
		this->Caches.push_back(cache);
	}else{
		if ( ! this->Ms.empty() ){
			const EigenMatrix S = this->Ms.back()->TransportManifold(step, M);
			const EigenMatrix Y = M.Gr - this->Ms.back()->TransportManifold(this->Ms.back()->Gr, M);
			EigenMatrix TildeH = EigenZero(M.P.size(), M.P.size());
			EigenMatrix C = EigenZero(M.P.size(), M.getDimension());
			for ( int i = 0; i < M.getDimension(); i++ ){
				const EigenMatrix new_vec = this->Ms.back()->TransportManifold(std::get<1>(this->EigenPairs[i]), M).reshaped<Eigen::RowMajor>();
				TildeH += std::get<0>(this->EigenPairs[i]) * new_vec * new_vec.transpose();
				C.col(i) = M.BasisSet[i].reshaped<Eigen::RowMajor>();
			}
			const EigenMatrix Cinv = C.completeOrthogonalDecomposition().pseudoInverse();
			const EigenMatrix Scol = S.reshaped<Eigen::RowMajor>();
			const EigenMatrix Ycol = Y.reshaped<Eigen::RowMajor>();
			const EigenMatrix TildeHS = TildeH * Cinv.transpose() * Cinv * Scol;
			const EigenMatrix term2 = TildeHS * Scol.transpose() * Cinv.transpose() * Cinv * TildeH / M.Inner(S, TildeHS.reshaped<Eigen::RowMajor>(M.P.rows(), M.P.cols()));
			const EigenMatrix term3 = Ycol * Ycol.transpose() / M.Inner(Y, S);
			EigenMatrix H = Cinv * ( TildeH - term2 + term3 ) * Cinv.transpose();
			this->EigenPairs = Diagonalize(H, M.BasisSet);
		}else this->EigenPairs = M.Hrm;
	}
	this->Ms.push_back(M.Clone());
}

static EigenMatrix Recursive_BFGS_Hess(
		std::unique_ptr<Manifold>* Ms,
		std::map<std::string, EigenMatrix>* Caches,
		int length,
		std::function<double (double)>& CautiousThreshold,
		EigenMatrix v){
	if ( length > 1 ){
		Manifold& M2 = *(Ms[length - 1]);
		Manifold& M1 = *(Ms[length - 2]);

		const EigenMatrix& S = Caches[length - 1]["S"];
		const EigenMatrix& Y = Caches[length - 1]["Y"];
		const EigenMatrix& YoverYS = Caches[length - 1]["YoverYS"];
		const EigenMatrix& HSoverSHS = Caches[length - 1]["HSoverSHS"];

		const EigenMatrix TV = M2.TransportManifold(v, M1);
		const EigenMatrix HTV = Recursive_BFGS_Hess(Ms, Caches, length - 1, CautiousThreshold, TV);
		const EigenMatrix Part1 = M1.TransportManifold(HTV, M2);

		const double score = M2.Inner(Y, S) / M2.Inner(S, S);
		const double threshold = CautiousThreshold(std::sqrt(M1.Inner(M1.Gr, M1.Gr)));
		const bool update = score > threshold;
		if ( ! update ) return Part1;

		const EigenMatrix Part2 = M2.Inner(Y, v) * YoverYS;
		const EigenMatrix Part3 = M2.Inner(S, Part1) * HSoverSHS;
		const EigenMatrix Total = Part1 + Part2 - Part3;
		return Total;
	}else return (*Ms)->Hr(v);
}

EigenMatrix BroydenFletcherGoldfarbShanno::HessianMatrixFree(EigenMatrix v){
	return Recursive_BFGS_Hess(
		this->Ms.data(), this->Caches.data(),
		this->Caches.size(), this->CautiousThreshold, v
	);
}

/*
void BroydenFletcherGoldfarbShanno(Manifold& M1, Manifold& M2, EigenMatrix step1){
	const EigenMatrix S = M1.TransportManifold(step1, M2.P);
	const EigenMatrix Y = M2.Gr - M1.TransportManifold(M1.Gr, M2.P);
	const EigenMatrix YoverYS = Y / M2.Inner(Y, S);
	const EigenMatrix HS = M1.TransportManifold(M1.Hr(step1), M2.P);
	const EigenMatrix HSoverSHS = HS / M2.Inner(S, HS);
	if ( M1.MatrixFree ){
		M2.Hr = [&M1, &M2, S, Y, YoverYS, HSoverSHS](EigenMatrix v){
			const EigenMatrix tmp = M2.TransportManifold(v, M1.P);
			const EigenMatrix H1tmp = M1.Hr(tmp);
			const EigenMatrix Hv1 = M1.TransportManifold(H1tmp, M2.P);
			const EigenMatrix Hv2 = M2.Inner(Y, v) * YoverYS;
			const EigenMatrix Hv3 = M2.Inner(S, Hv1) * HSoverSHS;
			const EigenMatrix Hv = Hv1 + Hv2 - Hv3;
			return Hv;
		};
	}else{
		EigenMatrix TildeH = EigenZero(M1.P.size(), M1.P.size());
		EigenMatrix C = EigenZero(M1.P.size(), M1.getDimension());
		for ( int i = 0; i < M1.getDimension(); i++ ){
			const EigenMatrix new_vec = M1.TransportManifold(std::get<1>(M1.Hrm[i]), M2.P).reshaped<Eigen::RowMajor>();
			TildeH += std::get<0>(M1.Hrm[i]) * new_vec * new_vec.transpose();
			C.col(i) = M2.BasisSet[i].reshaped<Eigen::RowMajor>();
		}
		const EigenMatrix Cinv = C.completeOrthogonalDecomposition().pseudoInverse();
		const EigenMatrix Scol = S.reshaped<Eigen::RowMajor>();
		const EigenMatrix Ycol = Y.reshaped<Eigen::RowMajor>();
		const EigenMatrix TildeHS = TildeH * Scol;
		const EigenMatrix term2 = TildeHS * Scol.transpose() * Cinv.transpose() * Cinv * TildeH / M2.Inner(S, TildeHS.reshaped<Eigen::RowMajor>(M1.P.rows(), M1.P.cols()));
		const EigenMatrix term3 = Ycol * Ycol.transpose() / M2.Inner(Y, S);
		EigenMatrix H = Cinv * ( TildeH - term2 + term3 ) * Cinv.transpose();
		M2.Hrm = Diagonalize(H, M2.BasisSet);
		M2.Hr = [&hrm = M2.Hrm](EigenMatrix v){
			EigenMatrix Hv = EigenZero(v.rows(), v.cols());
			for ( auto [eigenvalue, eigenvector] : hrm ){
				Hv += eigenvalue * eigenvector.cwiseProduct(v).sum() * eigenvector;
			}
			return Hv;
		};
	}
}*/

#ifdef __PYTHON__
void Init_HessUpdate(pybind11::module_& m){
	pybind11::class_<HessUpdate>(m, "HessUpdate")
		.def_readwrite("Size", &HessUpdate::Size)
		//.def_readwrite("Ms", &HessUpdate::Ms) // Exposing this one may invite trouble.
		.def_readwrite("Caches", &HessUpdate::Caches)
		.def_readwrite("EigenPairs", &HessUpdate::EigenPairs)
		.def(pybind11::init<int>())
		.def("Append", &HessUpdate::Append)
		.def("AdmittedAppend", &HessUpdate::AdmittedAppend)
		.def("Hessian", &HessUpdate::Hessian)
		.def("HessianMatrixFree", &HessUpdate::HessianMatrixFree)
		.def("Clear", &HessUpdate::Clear);
	pybind11::class_<BroydenFletcherGoldfarbShanno, HessUpdate>(m, "BroydenFletcherGoldfarbShanno")
		.def(pybind11::init<int>());
}
#endif
