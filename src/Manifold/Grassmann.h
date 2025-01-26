#include "Manifold.h"

class Grassmann: public Manifold{ public:
	EigenMatrix Projector;

	Grassmann(EigenMatrix p, bool hess_transport_matrix);

	int getDimension() override;
	double Inner(EigenMatrix X, EigenMatrix Y) override;

	EigenMatrix Exponential(EigenMatrix X) override;
	EigenMatrix Logarithm(EigenMatrix q) override;

	EigenMatrix TangentProjection(EigenMatrix A) override;
	EigenMatrix TangentPurification(EigenMatrix A) override;

	EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) override;
	EigenMatrix TransportManifold(EigenMatrix X, EigenMatrix q) override;

	void Update(EigenMatrix p, bool purify) override;
	void getGradient() override;
	void getHessian() override;
};
