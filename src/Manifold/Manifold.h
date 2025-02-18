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
	virtual int getDimension() const;
	virtual double Inner(EigenMatrix X, EigenMatrix Y) const;
	void getBasisSet();
	void getHessianMatrix();

	virtual EigenMatrix Exponential(EigenMatrix X) const;
	virtual EigenMatrix Logarithm(EigenMatrix q) const;

	virtual EigenMatrix TangentProjection(EigenMatrix A) const;
	virtual EigenMatrix TangentPurification(EigenMatrix A) const;

	virtual EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) const;
	virtual EigenMatrix TransportManifold(EigenMatrix X, EigenMatrix q) const;

	virtual void Update(EigenMatrix p, bool purify);
	virtual void getGradient();
	virtual void getHessian();

	~Manifold() = default;
	virtual std::unique_ptr<Manifold> Clone() const;
};

std::vector<std::tuple<double, EigenMatrix>> Diagonalize(
		EigenMatrix& A, std::vector<EigenMatrix>& basis_set);
