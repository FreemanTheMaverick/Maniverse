#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Dense>
#include <cmath>
#include <tuple>
#include <vector>
#include <functional>
#include <string>
#include <cassert>

#include "../Macro.h"
#include "../Manifold/Manifold.h"

#include <iostream>


void BroydenFletcherGoldfarbShanno(Manifold& M1, Manifold& M2, EigenMatrix step1){
	const EigenMatrix S = M1.TransportManifold(step1, M2.P);
	const EigenMatrix Y = M2.Gr - M1.TransportManifold(M1.Gr, M2.P);
	const EigenMatrix YoverYS = Y / M2.Inner(Y, S);
	const EigenMatrix tmp = M1.TransportManifold(M1.Hr(M2.TransportManifold(S, M1.P)), M2.P);
	const EigenMatrix HSoverSHS = tmp / M2.Inner(S, tmp);
	if ( M1.MatrixFree ){
		M2.Hr = [M1, &M2, S, Y, YoverYS, HSoverSHS](EigenMatrix v){
			const EigenMatrix Hv1 = M1.TransportManifold(M1.Hr(M2.TransportManifold(v, M1.P)), M2.P);
			const EigenMatrix Hv2 = M2.Inner(Y, v) * YoverYS;
			const EigenMatrix Hv3 = M2.Inner(S, Hv1) * HSoverSHS;
			return Hv1 + Hv2 - Hv3;
		};
	}else{
		EigenMatrix TildeH = EigenZero(M1.P.size(), M1.P.size());
		std::vector<EigenMatrix> basis_set(M1.getDimension(), EigenZero(M1.P.size(), M1.P.size()));
		EigenMatrix C = EigenZero(M1.P.size(), M1.getDimension());
		for ( int i = 0; i < M1.getDimension(); i++ ){
			basis_set[i] =  M1.TransportManifold(std::get<1>(M1.Hrm[i]), M2.P);
			C.col(i) = basis_set[i].reshaped<Eigen::RowMajor>();
			TildeH += std::get<0>(M1.Hrm[i]) * C.col(i) * C.col(i).transpose();
		}
		const EigenMatrix Cinv = C.completeOrthogonalDecomposition().pseudoInverse();
		const EigenMatrix Scol = S.reshaped<Eigen::RowMajor>();
		const EigenMatrix Ycol = Y.reshaped<Eigen::RowMajor>();
		const EigenMatrix TildeHS = TildeH * Scol;
		const EigenMatrix InnerM = Cinv.transpose() * Cinv;
		const EigenMatrix term2 = TildeHS * Scol.transpose() * InnerM * TildeH / M2.Inner(S, TildeHS.reshaped<Eigen::RowMajor>(M1.P.rows(), M1.P.cols()));
		const EigenMatrix term3 = Ycol * Ycol.transpose() / M2.Inner(Y, S);
		EigenMatrix H = Cinv * ( TildeH - term2 + term3 ) * C;
		M2.Hrm = Diagonalize(H, basis_set);
		M2.Hr = [&hrm = M2.Hrm](EigenMatrix v){
			EigenMatrix Hv = EigenZero(v.rows(), v.cols());
			for ( auto [eigenvalue, eigenvector] : hrm ){
				Hv += eigenvalue * eigenvector.cwiseProduct(v).sum() * eigenvector;
			}
			return Hv;
		};
	}
}

void Init_HessUpdate(pybind11::module_& m){
	m.def("BroydenFletcherGoldfarbShanno", &BroydenFletcherGoldfarbShanno);
}
