namespace Maniverse{

typedef std::function<
			std::tuple<
				double,
				std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>)
		> FixedPointFunc;

bool Anderson(
		FixedPointFunc& func,
		std::tuple<double, double, double> tol,
		double beta, int max_mem, int max_iter,
		double& L, Iterate& M, int output);

}
