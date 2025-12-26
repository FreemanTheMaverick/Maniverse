namespace Maniverse{

typedef std::function<
			std::tuple<
				double,
				std::vector<EigenMatrix>,
				std::vector<std::function<EigenMatrix (EigenMatrix)>>
			> (std::vector<EigenMatrix>, int)
		> UnpreconSecondFunc;

typedef std::function<
			std::tuple<
				double,
				std::vector<EigenMatrix>,
				std::vector<std::function<EigenMatrix (EigenMatrix)>>,
				std::vector<std::function<EigenMatrix (EigenMatrix)>>
			> (std::vector<EigenMatrix>, int)
		> PreconFunc;

class TruncatedConjugateGradient{ public:
	Iterate* M; // For inner product and tangent projection.
	bool Preconditioned;
	bool Verbose;
	bool ShowTarget;
	double Radius;
	std::function<bool (double, double, double, double)> Tolerance;
	std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> Sequence; // Step size, S, P.
	TruncatedConjugateGradient(){};
	TruncatedConjugateGradient(
			Iterate* m, bool preconditioned, bool verbose, bool showtarget
	): M(m), Preconditioned(preconditioned), Verbose(verbose), ShowTarget(showtarget){};
	void Run();
	std::tuple<double, EigenMatrix> Find(); // Step size, S.
};

template <typename FuncType>
bool TruncatedNewton(
		FuncType& func,
		TrustRegion& tr,
		std::tuple<double, double, double> tol,
		double tcg_tol, int max_iter,
		double& L, Iterate& M, int output);

extern template bool TruncatedNewton(
		UnpreconSecondFunc& func,
		TrustRegion& tr,
		std::tuple<double, double, double> tol,
		double tcg_tol, int max_iter,
		double& L, Iterate& M, int output);

extern template bool TruncatedNewton(
		PreconFunc& func,
		TrustRegion& tr,
		std::tuple<double, double, double> tol,
		double tcg_tol, int max_iter,
		double& L, Iterate& M, int output);

}
