#pragma once

namespace Maniverse{

class TrustRegion{ public:
	double R0;
	double RhoThreshold;
	std::function<double (double, double, double)> Update;
	TrustRegion();
};

}
