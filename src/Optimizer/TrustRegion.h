class TrustRegionSetting{ public:
	double R0;
	double RhoThreshold;
	std::function<double (double, double, double)> Update;
	TrustRegionSetting();
};

#define UnpreconFuncType\
		std::function<\
			std::tuple<\
				double,\
				std::vector<EigenMatrix>,\
				std::vector<std::function<EigenMatrix (EigenMatrix)>>\
			> (std::vector<EigenMatrix>, int)\
		>

#define PreconFuncType\
		std::function<\
			std::tuple<\
				double,\
				std::vector<EigenMatrix>,\
				std::vector<std::function<EigenMatrix (EigenMatrix)>>,\
				std::vector<std::tuple<\
					std::function<EigenMatrix (EigenMatrix)>,\
					std::function<EigenMatrix (EigenMatrix)>,\
					std::function<EigenMatrix (EigenMatrix)>\
				>>\
			> (std::vector<EigenMatrix>, int)\
		>

bool TrustRegion(
		UnpreconFuncType& func,
		TrustRegionSetting& tr_setting,
		std::tuple<double, double, double> tol,
		double tcg_tol,
		int recalc_hess, int max_iter,
		double& L, Iterate& M, int output);

bool TrustRegion(
		PreconFuncType& func,
		TrustRegionSetting& tr_setting,
		std::tuple<double, double, double> tol,
		double tcg_tol,
		int recalc_hess, int max_iter,
		double& L, Iterate& M, int output);
