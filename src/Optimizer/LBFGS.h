namespace Maniverse{

#define UnpreconFirstFunc\
		std::function<\
			std::tuple<\
				double,\
				std::vector<EigenMatrix>\
			> (std::vector<EigenMatrix>, int)\
		>

#define PreconFirstFunc\
		std::function<\
			std::tuple<\
				double,\
				std::vector<EigenMatrix>,\
				std::vector<std::function<EigenMatrix (EigenMatrix)>>,\
				std::vector<std::function<EigenMatrix (EigenMatrix)>>\
			> (std::vector<EigenMatrix>, int)\
		>
		// Preconditioners of S
		// Inverse Preconditioner of S
		// Preconditioner of G

template <typename FuncType>
bool LBFGS(
		FuncType& func,
		std::tuple<double, double, double> tol,
		int max_mem, int max_iter,
		double& L, Iterate& M, int output);

extern template bool LBFGS(
		UnpreconFirstFunc& func,
		std::tuple<double, double, double> tol,
		int max_iter, int max_mem,
		double& L, Iterate& M, int output);

extern template bool LBFGS(
		PreconFirstFunc& func,
		std::tuple<double, double, double> tol,
		int max_iter, int max_mem,
		double& L, Iterate& M, int output);

}
