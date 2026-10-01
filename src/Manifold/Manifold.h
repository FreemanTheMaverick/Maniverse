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

class Function{ public:
	virtual void Calculate(std::vector<Eigen::MatrixXd> P, std::vector<int> derivative);
	double Value = 0;
	std::vector<Eigen::MatrixXd> Gradient;
	virtual std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> X) const;
	virtual std::vector<Eigen::MatrixXd> Preconditioner(std::vector<Eigen::MatrixXd> X) const;
	virtual std::vector<Eigen::MatrixXd> PreconditionerInv(std::vector<Eigen::MatrixXd> X) const;
	virtual std::vector<Eigen::MatrixXd> PreconditionerSqrt(std::vector<Eigen::MatrixXd> X) const;
	virtual std::vector<Eigen::MatrixXd> PreconditionerInvSqrt(std::vector<Eigen::MatrixXd> X) const;
};

class Constraint{ public:
	int TotalSize = 0;
	std::vector<std::array<int, 3>> BlockParameters;
	Function* Func;
	std::vector<std::shared_ptr<Manifold>> Manifolds;
	Constraint(int total_size, std::vector<std::array<int, 3>> block_parameters, Function& func, std::vector<std::shared_ptr<Manifold>> manifolds);
	double Lambda = 0;
	Eigen::VectorXd Gradient;
	void setGradient();
	std::vector<Eigen::MatrixXd> getGradient() const;
	Eigen::VectorXd Hessian(Eigen::VectorXd X) const;
};

class Iterate{ public:
	Eigen::VectorXd Point;
	void setPoint(std::vector<Eigen::MatrixXd> ps, bool purify);
	std::vector<Eigen::MatrixXd> getPoint() const;

	// Total value of the (augmented) Lagrangian function
	std::vector<std::shared_ptr<Manifold>> Manifolds;
	void Calculate(std::vector<Eigen::MatrixXd> P, std::vector<int> derivatives);
	double Value;
	Eigen::VectorXd Gradient;
	void setGradient();
	std::vector<Eigen::MatrixXd> getGradient() const;
	Eigen::VectorXd Hessian(Eigen::VectorXd Xvec) const;
	Eigen::VectorXd Preconditioner(Eigen::VectorXd Xvec) const;
	Eigen::VectorXd PreconditionerInv(Eigen::VectorXd Xvec) const;
	Eigen::VectorXd PreconditionerSqrt(Eigen::VectorXd Xvec) const;
	Eigen::VectorXd PreconditionerInvSqrt(Eigen::VectorXd Xvec) const;

	// Objective function and its gradient and Hessian
	Function* Objective;
	Eigen::VectorXd ObjectiveGradient;
	void setObjectiveGradient();
	std::vector<Eigen::MatrixXd> getObjectiveGradient() const;
	Eigen::VectorXd ObjectiveHessian(Eigen::VectorXd Xvec) const;

	// Constraints
	std::vector<Constraint> Constraints;
	std::vector<double> calcLambda() const;
	void setLambda(std::vector<double> lambda);
	std::vector<double> getLambda() const;
	double Rho = 0;
	Eigen::VectorXd ConstraintProjection(Eigen::VectorXd A) const;

	int TotalSize = 0;
	std::vector<std::array<int, 3>> BlockParameters;

	Iterate(Function& objective, std::vector<std::shared_ptr<Manifold>> manifolds, std::vector<Function*> cons_funcs = {});

	// Manifold utilities
	std::string getName() const;
	int getDimension() const;

	double Inner(Eigen::VectorXd X, Eigen::VectorXd Y) const;
	Eigen::VectorXd Retract(Eigen::VectorXd X) const;
	Eigen::VectorXd InverseRetract(Iterate& N) const;
	Eigen::VectorXd TransportTangent(Eigen::VectorXd X, Eigen::VectorXd Y) const;
	Eigen::VectorXd TransportManifold(Eigen::VectorXd A, Iterate& N) const;

	Eigen::VectorXd TangentProjection(Eigen::VectorXd A) const;
	Eigen::VectorXd TangentPurification(Eigen::VectorXd A) const;
 
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
