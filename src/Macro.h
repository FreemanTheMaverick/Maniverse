#define EigenArray Eigen::ArrayXd
#define EigenVector Eigen::VectorXd
#define EigenDiagonal Eigen::DiagonalMatrix<double, -1, -1>
#define EigenMatrix Eigen::MatrixXd
#define EigenZero Eigen::MatrixXd::Zero
#define EigenOne Eigen::MatrixXd::Identity

#define Diag(X) (X).diagonal().asDiagonal()
#define Dot(X, Y) ( (X).transpose() * (Y) ).trace()

#define __Not_Implemented__\
	std::string func_name = __func__;\
	std::string class_name = typeid(*this).name();\
	throw std::runtime_error(func_name + " for " + class_name + " is not implemented!");

#define __True_False__(x) ( x ? "True" : "False" )

#define __now__ std::chrono::high_resolution_clock::now()
#define __duration__(start, end) std::chrono::duration<double>(end - start).count()
