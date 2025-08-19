#define UnpreconFuncType\
		std::function<\
			std::tuple<\
				double,\
				std::vector<EigenMatrix>\
			> (std::vector<EigenMatrix>, int)\
		>

#define PreconFuncType\
		std::function<\
			std::tuple<\
				double,\
				std::vector<EigenMatrix>,\
				std::vector<std::tuple<\
					std::function<EigenMatrix (EigenMatrix)>,\
					std::function<EigenMatrix (EigenMatrix)>,\
					std::function<EigenMatrix (EigenMatrix)>\
				>>\
			> (std::vector<EigenMatrix>, int)\
		>
		// Preconditioners of S
		// Inverse Preconditioner of S
		// Preconditioner of G

bool LBFGS(
		UnpreconFuncType& func,
		std::tuple<double, double, double> tol,
		int max_iter, int max_mem,
		double& L, Iterate& M, int output);

bool LBFGS(
		PreconFuncType& func,
		std::tuple<double, double, double> tol,
		int max_iter, int max_mem,
		double& L, Iterate& M, int output);
