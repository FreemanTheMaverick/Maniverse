#pragma once

namespace Maniverse{

#define __Check_Log_Map__\
	if ( typeid(N) != typeid(*this) )\
		throw std::runtime_error("The point to logarithm map is not in " + std::string(typeid(*this).name()) + "but in " + std::string(typeid(N).name()) + "!");

#define __Check_Vec_Transport__\
	if ( typeid(N) != typeid(*this) )\
		throw std::runtime_error("The destination of vector transport is not in " + std::string(typeid(*this).name()) + "but in " + std::string(typeid(N).name()) + "!");

[[maybe_unused]] static bool CompareString(std::string given, std::vector<std::string> strings){
	for ( std::string string : strings ) if ( string == given ) return 1;
	return 0;
}

#define __Check_Geodesic__(...)\
	if ( ! CompareString(this->Geodesic, {__VA_ARGS__}) ) throw std::runtime_error("Unimplemented geodesic type for " + std::string(typeid(*this).name()) + "!");

#define __Check_Geodesic_Func__\
	throw std::runtime_error("Currently " + this->Geodesic + " " + std::string(__func__) + " on " + std::string(typeid(*this).name()) + " is not supported!");

class Manifold{ public:
	std::string Name;
	std::string Geodesic;

	EigenMatrix P;
	EigenMatrix Ge;
	EigenMatrix Gr;

	std::vector<EigenMatrix> BasisSet;

	Manifold(EigenMatrix p, std::string geodesic);
	virtual int getDimension() const;
	virtual double Inner(EigenMatrix X, EigenMatrix Y) const;
	void getBasisSet();
	void getHessianMatrix();

	virtual EigenMatrix Retract(EigenMatrix X) const;
	virtual EigenMatrix InverseRetract(Manifold& N) const;
	virtual EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) const;
	virtual EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const;

	virtual EigenMatrix TangentProjection(EigenMatrix A) const;
	virtual EigenMatrix TangentPurification(EigenMatrix A) const;

	virtual void setPoint(EigenMatrix p, bool purify);

	virtual void getGradient();
	virtual EigenMatrix getHessian(EigenMatrix HeX, EigenMatrix X, bool weingarten) const;

	virtual ~Manifold() = default;
	virtual std::shared_ptr<Manifold> Share() const;
};

class Objective{ public:
	virtual void Calculate(std::vector<EigenMatrix> P, std::vector<int> derivative);
	double Value = 0;
	std::vector<EigenMatrix> Gradient = {};
	virtual std::vector<EigenMatrix> Hessian(std::vector<EigenMatrix> X) const;
	virtual std::vector<EigenMatrix> Preconditioner(std::vector<EigenMatrix> X) const;
	virtual std::vector<EigenMatrix> PreconditionerSqrt(std::vector<EigenMatrix> X) const;
	virtual std::vector<EigenMatrix> PreconditionerInvSqrt(std::vector<EigenMatrix> X) const;
};

class Iterate{ public:
	std::vector<std::shared_ptr<Manifold>> Ms;
	Objective* Func;
	EigenVector Point;
	EigenVector Gradient;
	EigenVector Hessian(EigenVector X) const;
	EigenVector Preconditioner(EigenVector X) const;
	EigenVector PreconditionerSqrt(EigenVector X) const;
	EigenVector PreconditionerInvSqrt(EigenVector X) const;

	int TotalSize;
	bool MatrixFree;
	std::vector<EigenMatrix> BasisSet;
	std::vector<std::tuple<double, EigenMatrix>> HessianMatrix;

	std::vector<std::tuple<int, int, int>> BlockParameters;

	Iterate(Objective& func, std::vector<std::shared_ptr<Manifold>> Ms, bool matrix_free);
	Iterate(const Iterate& another_iterate);

	std::string getName() const;
	int getDimension() const;
	double Inner(EigenVector X, EigenVector Y) const;

	EigenVector Retract(EigenVector X) const;
	EigenVector InverseRetract(Iterate& N) const;
	EigenVector TransportTangent(EigenVector X, EigenVector Y) const;
	EigenVector TransportManifold(EigenVector A, Iterate& N) const;

	EigenVector TangentProjection(EigenVector A) const;
	EigenVector TangentPurification(EigenVector A) const;
 
	void setPoint(std::vector<EigenMatrix> ps, bool purify);
	void setGradient();

	std::vector<EigenMatrix> getPoint() const;
	std::vector<EigenMatrix> getGradient() const;
	
	void getBasisSet();
	void getHessianMatrix();
};

#define GetBlock(mat, iM, BlockParameters)\
	Eigen::Map<const EigenMatrix>(\
			mat.data() + std::get<0>(BlockParameters[iM]),\
			std::get<1>(BlockParameters[iM]),\
			std::get<2>(BlockParameters[iM])\
	)

#define SetBlock(mat, iM, BlockParameters)\
	Eigen::Map<EigenMatrix> _##mat##_##iM##_(\
			mat.data() + std::get<0>(BlockParameters[iM]),\
			std::get<1>(BlockParameters[iM]),\
			std::get<2>(BlockParameters[iM])\
	); _##mat##_##iM##_

#define AssembleBlock(big_mat, mat_vec, BlockParameters){\
	for ( int _imat_ = 0; _imat_ < (int)mat_vec.size(); _imat_++ ){\
		SetBlock(big_mat, _imat_, BlockParameters) = mat_vec[_imat_];\
	}\
}

#define DecoupleBlock(big_mat, mat_vec, BlockParameters){\
	for ( int _imat_ = 0; _imat_ < (int)mat_vec.size(); _imat_++ )\
		mat_vec[_imat_] = GetBlock(big_mat, _imat_, BlockParameters);\
}

}
