class TruncatedConjugateGradient{ public:
	Manifold* M;
	std::function<EigenMatrix (EigenMatrix)>* Func;
	bool Verbose;
	bool ShowTarget;
	double Radius;
	std::function<bool (double, double, double, double)> Tolerance;
	std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> Sequence; // Step size, S, P.
	TruncatedConjugateGradient(){};
	TruncatedConjugateGradient(
			Manifold* m, std::function<EigenMatrix (EigenMatrix)>* func,
			bool verbose, bool showtarget
	): M(m), Func(func), Verbose(verbose), ShowTarget(showtarget){};
	void Run();
	std::tuple<double, EigenMatrix> Find(); // Step size, S.
};
