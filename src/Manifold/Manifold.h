#pragma once

#include <Eigen/Dense>
#include <typeinfo>
#include <string>
#include <vector>
#include <tuple>
#include <memory>

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

	Eigen::MatrixXd P;
	Eigen::MatrixXd Ge;
	Eigen::MatrixXd Gr;

	std::vector<Eigen::MatrixXd> BasisSet;

	Manifold(Eigen::MatrixXd p, std::string geodesic);
	virtual int getDimension() const;
	virtual double Inner(Eigen::MatrixXd X, Eigen::MatrixXd Y) const;
	void getBasisSet();
	void getHessianMatrix();

	virtual Eigen::MatrixXd Retract(Eigen::MatrixXd X) const;
	virtual Eigen::MatrixXd InverseRetract(Manifold& N) const;
	virtual Eigen::MatrixXd TransportTangent(Eigen::MatrixXd X, Eigen::MatrixXd Y) const;
	virtual Eigen::MatrixXd TransportManifold(Eigen::MatrixXd X, Manifold& N) const;

	virtual Eigen::MatrixXd TangentProjection(Eigen::MatrixXd A) const;
	virtual Eigen::MatrixXd TangentPurification(Eigen::MatrixXd A) const;

	virtual void setPoint(Eigen::MatrixXd p, bool purify);

	virtual void getGradient();
	virtual Eigen::MatrixXd getHessian(Eigen::MatrixXd HeX, Eigen::MatrixXd X, bool weingarten) const;

	virtual ~Manifold() = default;
	virtual std::unique_ptr<Manifold> Clone() const;
	virtual std::shared_ptr<Manifold> Share() const;
};

class Objective{ public:
	virtual void Calculate(std::vector<Eigen::MatrixXd> P, std::vector<int> derivative);
	double Value = 0;
	std::vector<Eigen::MatrixXd> Gradient;
	virtual std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> X) const;
	virtual std::vector<Eigen::MatrixXd> Preconditioner(std::vector<Eigen::MatrixXd> X) const;
	virtual std::vector<Eigen::MatrixXd> PreconditionerSqrt(std::vector<Eigen::MatrixXd> X) const;
	virtual std::vector<Eigen::MatrixXd> PreconditionerInvSqrt(std::vector<Eigen::MatrixXd> X) const;
	std::vector<double> Lambda;
	double Rho;
	std::vector<double> Constraint_Value;
	std::vector<std::vector<Eigen::MatrixXd>> Constraint_Gradient;
};

class Iterate{ public:
	std::vector<std::shared_ptr<Manifold>> Ms;
	Objective* Func;
	Eigen::VectorXd Point;
	Eigen::VectorXd Gradient;
	Eigen::VectorXd Hessian(Eigen::VectorXd X) const;
	Eigen::VectorXd Preconditioner(Eigen::VectorXd X) const;
	Eigen::VectorXd PreconditionerSqrt(Eigen::VectorXd X) const;
	Eigen::VectorXd PreconditionerInvSqrt(Eigen::VectorXd X) const;

	std::vector<std::vector<std::unique_ptr<Manifold>>> Constraints;
	std::vector<Eigen::VectorXd> Constraint_Gradient;

	int TotalSize;
	std::vector<std::tuple<int, int, int>> BlockParameters;

	Iterate(Objective& func, std::vector<std::shared_ptr<Manifold>> Ms);

	std::string getName() const;
	int getDimension() const;
	double Inner(Eigen::VectorXd X, Eigen::VectorXd Y) const;

	Eigen::VectorXd Retract(Eigen::VectorXd X) const;
	Eigen::VectorXd InverseRetract(Iterate& N) const;
	Eigen::VectorXd TransportTangent(Eigen::VectorXd X, Eigen::VectorXd Y) const;
	Eigen::VectorXd TransportManifold(Eigen::VectorXd A, Iterate& N) const;

	Eigen::VectorXd TangentProjection(Eigen::VectorXd A) const;
	Eigen::VectorXd TangentPurification(Eigen::VectorXd A) const;
 
	void setPoint(std::vector<Eigen::MatrixXd> ps, bool purify);
	void setGradient();

	std::vector<Eigen::MatrixXd> getPoint() const;
	std::vector<Eigen::MatrixXd> getGradient() const;
};

#define GetBlock(mat, iM, BlockParameters)\
	Eigen::Map<const Eigen::MatrixXd>(\
			mat.data() + std::get<0>(BlockParameters[iM]),\
			std::get<1>(BlockParameters[iM]),\
			std::get<2>(BlockParameters[iM])\
	)

#define SetBlock(mat, iM, BlockParameters)\
	Eigen::Map<Eigen::MatrixXd> _##mat##_##iM##_(\
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
