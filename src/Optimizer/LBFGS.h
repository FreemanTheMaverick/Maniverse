bool LBFGS(
		std::function<
			std::tuple<
				double,
				std::vector<EigenMatrix>
			> (std::vector<EigenMatrix>, int)
		>& func,
		std::tuple<double, double, double> tol,
		int max_iter, int max_mem,
		double& L, Iterate& M, int output);
