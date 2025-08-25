namespace Maniverse{

class TruncatedConjugateGradient{ public:
	Iterate* M; // For inner product and tangent projection.
	std::function<EigenMatrix (EigenMatrix)>* Func;
	bool Preconditioned;
	bool Verbose;
	bool ShowTarget;
	double Radius;
	std::function<bool (double, double, double, double)> Tolerance;
	std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> Sequence; // Step size, S, P.
	TruncatedConjugateGradient(){};
	TruncatedConjugateGradient(
			Iterate* m, std::function<EigenMatrix (EigenMatrix)>* func,
			bool preconditioned, bool verbose, bool showtarget
	): M(m), Func(func), Preconditioned(preconditioned), Verbose(verbose), ShowTarget(showtarget){};
	void Run();
	std::tuple<double, EigenMatrix> Find(); // Step size, S.
};

}
