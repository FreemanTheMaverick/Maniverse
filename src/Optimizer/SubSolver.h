std::vector<std::tuple<double, EigenMatrix, EigenMatrix>> TruncatedConjugateGradient(
		Manifold& M, double R,
		std::tuple<double, double, double> tol, int output);

EigenMatrix RestartTCG(Manifold& M, std::vector<std::tuple<double, EigenMatrix, EigenMatrix>>& Vs, double R);
