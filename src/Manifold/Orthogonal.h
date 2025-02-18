#include "Manifold.h"

class Orthogonal: public Manifold{ public:
	Orthogonal(EigenMatrix p, bool matrix_free);

	int getDimension() const override;
	double Inner(EigenMatrix X, EigenMatrix Y) const override;

	EigenMatrix Exponential(EigenMatrix X) const override;
	EigenMatrix Logarithm(EigenMatrix q) const override;

	EigenMatrix TangentProjection(EigenMatrix A) const override;
	EigenMatrix TangentPurification(EigenMatrix A) const override;

	//EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) override;
	//EigenMatrix TransportManifold(EigenMatrix X, EigenMatrix q) override;

	void Update(EigenMatrix p, bool purify) override;
	void getGradient() override;
	void getHessian() override;

	std::unique_ptr<Manifold> Clone() const override;
};
