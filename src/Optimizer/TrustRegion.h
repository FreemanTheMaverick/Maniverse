bool TrustRegion(
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

bool TrustRegionMatrixFree(
		std::function<
			std::tuple<
				double,
				EigenMatrix,
				std::function<EigenMatrix (EigenMatrix)>
			> (EigenMatrix)
		>& func,
		std::tuple<double, double, double> tol,
		int max_iter,
		double& L, Manifold& M, int output);
