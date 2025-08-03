#ifdef PyManiverseIn
Init_Manifold(m);
Init_Iterate(m);

Init_Euclidean(m);
Init_TransRotInvPointCloud(m);
Init_RealSymmetric(m);

Init_Stiefel(m);
Init_Orthogonal(m);
Init_Grassmann(m);
Init_Flag(m);

Init_Simplex(m);
#endif


#ifdef PyManiverseOut
void Init_Manifold(pybind11::module_& m);
void Init_Iterate(pybind11::module_& m);

void Init_Euclidean(pybind11::module_& m);
void Init_TransRotInvPointCloud(pybind11::module_& m);
void Init_RealSymmetric(pybind11::module_& m);

void Init_Stiefel(pybind11::module_& m);
void Init_Orthogonal(pybind11::module_& m);
void Init_Grassmann(pybind11::module_& m);
void Init_Flag(pybind11::module_& m);

void Init_Simplex(pybind11::module_& m);
#endif
