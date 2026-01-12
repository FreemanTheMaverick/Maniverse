namespace Maniverse{

class TruncatedConjugateGradient{ public:
	Iterate* M; // For inner product and tangent projection.
	bool Verbose;
	bool ShowTarget;
	double Radius;
	std::function<bool (double, double, double, double)> Tolerance;
	std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> Sequence; // Step size, S, P.
	TruncatedConjugateGradient(){};
	TruncatedConjugateGradient(
			Iterate* m, bool verbose, bool showtarget
	): M(m), Verbose(verbose), ShowTarget(showtarget){};
	void Run();
	std::tuple<double, EigenMatrix> Find(); // Step size, S.
};

bool TruncatedNewton(
		Iterate& M,
		TrustRegion& tr,
		std::tuple<double, double, double> tol,
		double tcg_tol, int max_iter,
		int output);

}
