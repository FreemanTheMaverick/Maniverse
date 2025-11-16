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
		// Preconditioners of S
		// Inverse Preconditioner of S
		// Preconditioner of G

template <typename FuncType>
bool ArmijoBacktracking(
		FuncType& func,
		Iterate& M, EigenMatrix& S,
		double c1, double tau, int max_iter,
		double& L, std::vector<EigenMatrix>* Ge,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func1,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func2,
		int output);

extern template bool ArmijoBacktracking(
		UnpreconFirstFunc& func,
		Iterate& M, EigenMatrix& S,
		double c1, double tau, int max_iter,
		double& L, std::vector<EigenMatrix>* Ge,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func1,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func2,
		int output);

extern template bool ArmijoBacktracking(
		UnpreconSecondFunc& func,
		Iterate& M, EigenMatrix& S,
		double c1, double tau, int max_iter,
		double& L, std::vector<EigenMatrix>* Ge,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func1,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func2,
		int output);

extern template bool ArmijoBacktracking(
		PreconFunc& func,
		Iterate& M, EigenMatrix& S,
		double c1, double tau, int max_iter,
		double& L, std::vector<EigenMatrix>* Ge,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func1,
		[[maybe_unused]] std::vector<std::function<EigenMatrix (EigenMatrix)>>* func2,
		int output);

}
