#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <string>
#include <cassert>

#include "../Macro.h"
#include "../Manifold/Manifold.h"


void BroydenFletcherGoldfarbShanno(Manifold& M1, Manifold& M2, EigenMatrix step1){
	const EigenMatrix S = M1.TransportManifold(step1, M2.P);
	const EigenMatrix Y = M2.Gr - M1.TransportManifold(M1.Gr, M2.P);
	const EigenMatrix YoverYS = Y / M2.Inner(Y, S);
	const EigenMatrix tmp = M1.TransportManifold(M1.Hr(M2.TransportManifold(S, M1.P)), M2.P);
	const EigenMatrix HSoverSHS = tmp / M2.Inner(S, tmp);
	M2.Hr = [M1, &M2, S, Y, YoverYS, HSoverSHS](EigenMatrix v){
		const EigenMatrix Hv1 = M1.TransportManifold(M1.Hr(M2.TransportManifold(v, M1.P)), M2.P);
		const EigenMatrix Hv2 = M2.Inner(Y, v) * YoverYS;
		const EigenMatrix Hv3 = M2.Inner(S, Hv1) * HSoverSHS;
		return Hv1 + Hv2 + Hv3;
	};
}

void Init_HessUpdate(pybind11::module_& m){
	m.def("BroydenFletcherGoldfarbShanno", &BroydenFletcherGoldfarbShanno);
}
