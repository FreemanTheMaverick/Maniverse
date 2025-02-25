class HessUpdate{ public:
	int Size = 0;
	std::vector<std::unique_ptr<Manifold>> Ms;
	std::vector<std::map<std::string, EigenMatrix>> Caches; // For matrix-free
	std::vector<std::tuple<double, EigenMatrix>> EigenPairs; // For non-matrix-free
	HessUpdate(int n);
	void Append(Manifold& M, EigenMatrix Step);
	virtual void AdmittedAppend(Manifold& M, EigenMatrix Step);
	EigenMatrix Hessian(EigenMatrix v);
	virtual EigenMatrix HessianMatrixFree(EigenMatrix v);
	void Clear();
};

class BroydenFletcherGoldfarbShanno: public HessUpdate{ public:
	BroydenFletcherGoldfarbShanno(int n): HessUpdate(n){}
	void AdmittedAppend(Manifold& M, EigenMatrix Step) override;
	EigenMatrix HessianMatrixFree(EigenMatrix v) override;
};
