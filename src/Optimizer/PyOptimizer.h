#ifdef PyManiverseIn
Init_TrustRegion(m);
Init_SubSolver(m);
Init_HessUpdate(m);
Init_LBFGS(m);
#endif

#ifdef PyManiverseOut
void Init_TrustRegion(pybind11::module_& m);
void Init_SubSolver(pybind11::module_& m);
void Init_HessUpdate(pybind11::module_& m);
void Init_LBFGS(pybind11::module_& m);
#endif
