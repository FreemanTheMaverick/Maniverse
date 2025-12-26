#ifdef PyManiverseIn
Init_TrustRegion(m);
Init_TruncatedNewton(m);
Init_HessUpdate(m);
Init_LBFGS(m);
Init_Anderson(m);
#endif

#ifdef PyManiverseOut
void Init_TrustRegion(pybind11::module_& m);
void Init_TruncatedNewton(pybind11::module_& m);
void Init_HessUpdate(pybind11::module_& m);
void Init_LBFGS(pybind11::module_& m);
void Init_Anderson(pybind11::module_& m);
#endif
