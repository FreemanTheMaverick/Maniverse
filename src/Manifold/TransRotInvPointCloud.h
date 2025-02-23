#include "Manifold.h"

class TransRotInvPointCloud: public Manifold{ public:
	TransRotInvPointCloud(EigenMatrix p, bool matrix_free);

	int getDimension() const override;
	double Inner(EigenMatrix X, EigenMatrix Y) const override;

	EigenMatrix Exponential(EigenMatrix X) const override;
	EigenMatrix Logarithm(Manifold& N) const override;

	EigenMatrix TangentProjection(EigenMatrix A) const override;
	EigenMatrix TangentPurification(EigenMatrix A) const override;

	//EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) override const;
	EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const override;

	void Update(EigenMatrix p, bool purify) override;
	void getGradient() override;
	void getHessian() override;

	std::unique_ptr<Manifold> Clone() const override;
};
