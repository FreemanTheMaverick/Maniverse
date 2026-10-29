#ifdef PyManiverseIn
Init_LinearSolver(m);
Init_ConjugateGradient(m);
Init_MinRes(m);
#endif

#ifdef PyManiverseOut
void Init_LinearSolver(pybind11::module_& m);
void Init_ConjugateGradient(pybind11::module_& m);
void Init_MinRes(pybind11::module_& m);
#endif
