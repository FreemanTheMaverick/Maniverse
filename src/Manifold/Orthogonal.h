#include "Stiefel.h"

class Orthogonal: public Stiefel{ public:
	Orthogonal(EigenMatrix p, std::string geodesic = "POLAR");

	EigenMatrix Retract(EigenMatrix X) const override;
	EigenMatrix InverseRetract(Manifold& N) const override;

	std::unique_ptr<Manifold> Clone() const override;
};
