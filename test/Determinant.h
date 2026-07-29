#pragma once

#include <Eigen/Dense>
#include <vector>
#include <Maniverse/Manifold/Manifold.h>

class ObjDeterminant: public mv::Objective{ public:
	Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(10, 5);
	Eigen::MatrixXd C = Eigen::MatrixXd::Zero(10, 5);
	Eigen::MatrixXd C0tC = Eigen::MatrixXd::Zero(5, 5);
	Eigen::MatrixXd C0tCinv = Eigen::MatrixXd::Zero(5, 5);
	int rank = 0;

	// Rank-deficient
	double beta = 0;

	// Rank-deficient 1
	Eigen::VectorXd u0 = Eigen::VectorXd::Zero(10);
	Eigen::VectorXd v0 = Eigen::VectorXd::Zero(10);

	// Rank-deficient 2
	Eigen::MatrixXd U0 = Eigen::MatrixXd::Zero(10, 2);
	Eigen::MatrixXd V0 = Eigen::MatrixXd::Zero(10, 2);

	ObjDeterminant(Eigen::MatrixXd C0) : C0(C0){};

	void Calculate(std::vector<Eigen::MatrixXd> C_, std::vector<int> derivatives) override{
		C = C_[0];
		if ( std::count(derivatives.begin(), derivatives.end(), 0) ){
			C0tC = C0.transpose() * C;
			Value = C0tC.determinant();
			if ( derivatives.size() == 1 ) return;
		}
		Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(C0tC);
		rank = svd.rank();
		const Eigen::VectorXd S = svd.singularValues().head(rank);
		C0tCinv = svd.matrixV().leftCols(rank) * S.cwiseInverse().asDiagonal() * svd.matrixU().leftCols(rank).transpose();
		if ( rank == 5 ) Gradient = { Value * C0 * C0tCinv.transpose() };
		else{
			const double sing_prod = S.prod();
			const double detUV = ( svd.matrixU() * svd.matrixV() ).determinant();
			beta = sing_prod * detUV;
			if ( rank == 4 ){
				u0 = svd.matrixU().col(4);
				v0 = svd.matrixV().col(4);
				Gradient = { sing_prod * C0 * v0 * u0.transpose() };
			}else if ( rank == 3 ){
				U0 = svd.matrixU().rightCols(2);
				V0 = svd.matrixV().rightCols(2);
				Gradient = { Eigen::MatrixXd::Zero(10, 5) };
			}else Gradient = { Eigen::MatrixXd::Zero(10, 5) };
		}
	};

	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> X_) const override{
		const Eigen::MatrixXd X = X_[0];
		if ( rank == 5 ) return std::vector<Eigen::MatrixXd>{ Value * C0 * (
				C0tCinv.cwiseProduct( C0.transpose() * X ).sum() * C0tCinv.transpose()
				- C0tCinv.transpose() * X.transpose() * C0 * C0tCinv.transpose()
		) };
		if ( rank == 4 ) return std::vector<Eigen::MatrixXd>{ 2 * beta * (
				C0 * u0 * v0.transpose() * C0tCinv.cwiseProduct( C0.transpose() * X ).sum()
				- C0 * u0 * v0.transpose() * X.transpose() * C0 * C0tCinv.transpose()
				- u0 * v0.transpose() * C0 * X * C0 * C0tCinv.transpose()
		) };
		if ( rank == 3 ){
			Eigen::MatrixXd M = U0.transpose() * C0.transpose() * X * V0;
			M(0, 1) *= -1;
			M(1, 0) *= -1;
			std::swap(M(0, 0), M(1, 1));
			return std::vector<Eigen::MatrixXd>{ 2 * beta * C0 * U0 * M.transpose() * V0.transpose() };
		}
		return { Eigen::MatrixXd::Zero(10, 5) };
	};
};

class ObjDeterminants: public mv::Objective{ public:
	std::vector<ObjDeterminant> Funcs;
	ObjDeterminants(std::vector<Eigen::MatrixXd> C0s){
		for ( Eigen::MatrixXd& C0 : C0s ) Funcs.push_back(ObjDeterminant(C0));
	};
	void Calculate(std::vector<Eigen::MatrixXd> Cs_, std::vector<int> derivatives){
		Value = 0;
		Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(Cs_[0].rows(), Cs_[0].cols());
		for ( ObjDeterminant& func : Funcs ){
			func.Calculate(Cs_, derivatives);
			Value += func.Value;
			gradient += func.Gradient[0];
		}
		Gradient = { gradient };
	};
	std::vector<Eigen::MatrixXd> Hessian(std::vector<Eigen::MatrixXd> Xs_) const override{
		Eigen::MatrixXd HX = Eigen::MatrixXd::Zero(Xs_[0].rows(), Xs_[0].cols());
		for ( const ObjDeterminant& func : Funcs ) HX += func.Hessian(Xs_)[0];
		return { HX };
	};
};
