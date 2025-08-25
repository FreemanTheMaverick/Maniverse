namespace Maniverse{

class TrustRegionSetting{ public:
	double R0;
	double RhoThreshold;
	std::function<double (double, double, double)> Update;
	TrustRegionSetting();
};

#define UnpreconSecondFunc\
		std::function<\
			std::tuple<\
				double,\
				std::vector<EigenMatrix>,\
				std::vector<std::function<EigenMatrix (EigenMatrix)>>\
			> (std::vector<EigenMatrix>, int)\
		>

#define PreconSecondFunc\
		std::function<\
			std::tuple<\
				double,\
				std::vector<EigenMatrix>,\
				std::vector<std::function<EigenMatrix (EigenMatrix)>>,\
				std::vector<std::function<EigenMatrix (EigenMatrix)>>\
			> (std::vector<EigenMatrix>, int)\
		>

template <typename FuncType>
bool TrustRegion(
		FuncType& func,
		TrustRegionSetting& tr_setting,
		std::tuple<double, double, double> tol,
		double tcg_tol,
		int recalc_hess, int max_iter,
		double& L, Iterate& M, int output);

extern template bool TrustRegion(
		UnpreconSecondFunc& func,
		TrustRegionSetting& tr_setting,
		std::tuple<double, double, double> tol,
		double tcg_tol,
		int recalc_hess, int max_iter,
		double& L, Iterate& M, int output);

extern template bool TrustRegion(
		PreconSecondFunc& func,
		TrustRegionSetting& tr_setting,
		std::tuple<double, double, double> tol,
		double tcg_tol,
		int recalc_hess, int max_iter,
		double& L, Iterate& M, int output);

}
