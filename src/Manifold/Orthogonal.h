#include "Stiefel.h"

class Orthogonal: public Stiefel{ public:
	Orthogonal(EigenMatrix p);

	EigenMatrix Exponential(EigenMatrix X) const override;
	EigenMatrix Logarithm(Manifold& N) const override;

	EigenMatrix TangentPurification(EigenMatrix A) const override;
	EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) const override;
	EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const override;

	std::unique_ptr<Manifold> Clone() const override;
};
