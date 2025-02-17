bool TrustRegion(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix, int)
		>& func,
		std::tuple<double, double, double> tol,
		int recalc_hess, int max_iter,
		double& L, Manifold& M, int output);

bool TrustRegionRationalFunctionOptimization(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix, int)
		>& func,
		int order,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, Manifold& M, int output);
