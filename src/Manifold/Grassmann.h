#include "Manifold.h"

class Grassmann: public Manifold{ public:
	EigenMatrix Projector;
	mutable std::vector<std::tuple<EigenMatrix, EigenMatrix>> LogCache;
	mutable std::vector<std::tuple<EigenMatrix, EigenMatrix, EigenMatrix>> TransportTangentCache;

	Grassmann(EigenMatrix p);

	int getDimension() const override;
	double Inner(EigenMatrix X, EigenMatrix Y) const override;

	EigenMatrix Exponential(EigenMatrix X) const override;
	EigenMatrix Logarithm(Manifold& N) const override;

	EigenMatrix TangentProjection(EigenMatrix A) const override;
	EigenMatrix TangentPurification(EigenMatrix A) const override;

	EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) const override;
	EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const override;

	void setPoint(EigenMatrix p, bool purify) override;

	void getGradient() override;
	std::function<EigenMatrix (EigenMatrix)> getHessian(std::function<EigenMatrix (EigenMatrix)> h, bool weingarten) const override;

	std::unique_ptr<Manifold> Clone() const override;
};
