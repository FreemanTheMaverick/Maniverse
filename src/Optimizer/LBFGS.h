namespace Maniverse{

typedef std::function<
			std::tuple<
				double,
				std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>, int)
		> UnpreconFirstFunc;

typedef std::function<
			std::tuple<
				double,
				std::vector<EigenMatrix>,
				std::vector<std::function<EigenMatrix (EigenMatrix)>>,
				std::vector<std::function<EigenMatrix (EigenMatrix)>>
			> (std::vector<EigenMatrix>, int)
		> PreconFunc;
		// Preconditioners of S
		// Inverse Preconditioner of S
		// Preconditioner of G

template <typename FuncType>
bool LBFGS(
		FuncType& func,
		std::tuple<double, double, double> tol,
		int max_mem, int max_iter,
		double c1, double tau, int ls_max_iter,
		double& L, Iterate& M, int output);

extern template bool LBFGS(
		UnpreconFirstFunc& func,
		std::tuple<double, double, double> tol,
		int max_mem, int max_iter,
		double c1, double tau, int ls_max_iter,
		double& L, Iterate& M, int output);

extern template bool LBFGS(
		PreconFunc& func,
		std::tuple<double, double, double> tol,
		int max_mem, int max_iter,
		double c1, double tau, int ls_max_iter,
		double& L, Iterate& M, int output);

}
