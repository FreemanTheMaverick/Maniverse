#pragma once

class Manifold{ public:
	std::string Name;
	EigenMatrix P;
	EigenMatrix Ge;
	EigenMatrix Gr;
	bool MatrixFree;
	EigenMatrix Hem;
	std::vector<std::tuple<double, EigenMatrix>> Hrm;
	std::function<EigenMatrix (EigenMatrix)> He;
	std::function<EigenMatrix (EigenMatrix)> Hr;
	std::vector<EigenMatrix> BasisSet;

	Manifold(EigenMatrix p, bool matrix_free);
	virtual int getDimension();
	virtual double Inner(EigenMatrix X, EigenMatrix Y);
	void getBasisSet();
	void getHessianMatrix();

	virtual EigenMatrix Exponential(EigenMatrix X);
	virtual EigenMatrix Logarithm(EigenMatrix q);

	virtual EigenMatrix TangentProjection(EigenMatrix A);
	virtual EigenMatrix TangentPurification(EigenMatrix A);

	virtual EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y);
	virtual EigenMatrix TransportManifold(EigenMatrix X, EigenMatrix q);

	virtual void Update(EigenMatrix p, bool purify);
	virtual void getGradient();
	virtual void getHessian();
};
