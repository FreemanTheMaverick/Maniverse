namespace Maniverse{

bool LBFGS(
		Iterate& M,
		std::tuple<double, double, double> tol,
		int max_mem, int max_iter,
		double c1, double tau, int ls_max_iter,
		int output);

}
