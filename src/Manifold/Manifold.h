#pragma once

#define __Check_Log_Map__\
	if ( typeid(N) != typeid(*this) )\
		throw std::runtime_error("The point to logarithm map is not in " + std::string(typeid(*this).name()) + "but in " + std::string(typeid(N).name()) + "!");

#define __Check_Vec_Transport__\
	if ( typeid(N) != typeid(*this) )\
		throw std::runtime_error("The destination of vector transport is not in " + std::string(typeid(*this).name()) + "but in " + std::string(typeid(N).name()) + "!");

class Manifold{ public:
	std::string Name;
	EigenMatrix P;
	EigenMatrix Ge;
	EigenMatrix Gr;

	std::vector<EigenMatrix> BasisSet;

	Manifold(EigenMatrix p);
	virtual int getDimension() const;
	virtual double Inner(EigenMatrix X, EigenMatrix Y) const;
	void getBasisSet();
	void getHessianMatrix();

	virtual EigenMatrix Exponential(EigenMatrix X) const;
	virtual EigenMatrix Logarithm(Manifold& N) const;

	virtual EigenMatrix TangentProjection(EigenMatrix A) const;
	virtual EigenMatrix TangentPurification(EigenMatrix A) const;

	virtual EigenMatrix TransportTangent(EigenMatrix X, EigenMatrix Y) const;
	virtual EigenMatrix TransportManifold(EigenMatrix X, Manifold& N) const;

	virtual void setPoint(EigenMatrix p, bool purify);

	virtual void getGradient();
	virtual std::function<EigenMatrix (EigenMatrix)> getHessian(std::function<EigenMatrix (EigenMatrix)> He, bool weingarten) const;

	virtual ~Manifold() = default;
	virtual std::unique_ptr<Manifold> Clone() const;
};

std::vector<std::tuple<double, EigenMatrix>> Diagonalize(
		EigenMatrix& A, std::vector<EigenMatrix>& basis_set);

class Iterate{ public:
	std::vector<std::unique_ptr<Manifold>> Ms;
	EigenMatrix Point;
	EigenMatrix Gradient;
	std::function<EigenMatrix (EigenMatrix)> Hessian;

	bool MatrixFree;
	std::vector<EigenMatrix> BasisSet;
	std::vector<std::tuple<double, EigenMatrix>> HessianMatrix;

	std::vector<std::tuple<int, int, int, int>> BlockParameters;

	Iterate(std::vector<std::shared_ptr<Manifold>> Ms, bool matrix_free);
	Iterate(const Iterate& another_iterate);

	std::string getName() const;
	int getDimension() const;
	double Inner(EigenMatrix X, EigenMatrix Y) const;

	EigenMatrix Exponential(EigenMatrix X) const;

	EigenMatrix TangentProjection(EigenMatrix A) const;
	EigenMatrix TangentPurification(EigenMatrix A) const;
 
	EigenMatrix TransportManifold(EigenMatrix A, Iterate& N) const;

	void setPoint(std::vector<EigenMatrix> ps, bool purify);

	void setGradient(std::vector<EigenMatrix> gs);
	void setHessian(std::vector<std::function<EigenMatrix (EigenMatrix)>> hs);
	
	std::vector<EigenMatrix> getPoint() const;
	std::vector<EigenMatrix> getGradient() const;
	
	void getBasisSet();
	void getHessianMatrix();
};

#define GetBlock(mat, iM)\
	mat.block(\
			std::get<0>(BlockParameters[iM]),\
			std::get<1>(BlockParameters[iM]),\
			std::get<2>(BlockParameters[iM]),\
			std::get<3>(BlockParameters[iM])\
	)

#define AssemblyBlock(big_mat, mat_vec){\
	int _nrows_ = 0;\
	int _ncols_ = 0;\
	for ( EigenMatrix& mat : mat_vec ){\
		big_mat.block(\
				_nrows_, _ncols_,\
				mat.rows(), mat.cols()\
		) = mat;\
		_nrows_ += mat.rows();\
		_ncols_ += mat.cols();\
	}\
}

#define DecoupleBlock(big_mat, mat_vec){\
	int _nrows_ = 0;\
	int _ncols_ = 0;\
	for ( EigenMatrix& mat : mat_vec ){\
		mat = big_mat.block(\
				_nrows_, _ncols_,\
				mat.rows(), mat.cols()\
		);\
		_nrows_ += mat.rows();\
		_ncols_ += mat.cols();\
	}\
}
