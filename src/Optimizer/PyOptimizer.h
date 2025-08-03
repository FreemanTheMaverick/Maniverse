#ifdef PyManiverseIn
Init_TrustRegion(m);
Init_SubSolver(m);
Init_HessUpdate(m);
#endif

#ifdef PyManiverseOut
void Init_TrustRegion(pybind11::module_& m);
void Init_SubSolver(pybind11::module_& m);
void Init_HessUpdate(pybind11::module_& m);
#endif
