#pragma once

#define __Check_Log_Map__\
	if ( typeid(N) != typeid(*this) )\
		throw std::runtime_error("The point to logarithm map is not in " + std::string(typeid(*this).name()) + "but in " + std::string(typeid(N).name()) + "!");

#define __Check_Vec_Transport__\
	if ( typeid(N) != typeid(*this) )\
		throw std::runtime_error("The destination of vector transport is not in " + std::string(typeid(*this).name()) + "but in " + std::string(typeid(N).name()) + "!");

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
	virtual EigenMatrix Logarithm(Manifold& N) const;

	virtual EigenMatrix TangentProjection(EigenMatrix A) const;
	virtual EigenMatrix TangentPurification(EigenMatrix A) const;

	virtual EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) const;
	virtual EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const;

	virtual void Update(EigenMatrix p, bool purify);
	virtual void getGradient();
	virtual void getHessian();

	virtual ~Manifold() = default;
	virtual std::unique_ptr<Manifold> Clone() const;
};

std::vector<std::tuple<double, EigenMatrix>> Diagonalize(
		EigenMatrix& A, std::vector<EigenMatrix>& basis_set);
