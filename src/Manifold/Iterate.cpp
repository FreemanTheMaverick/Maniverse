#ifdef __PYTHON__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#endif

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <tuple>

#include "Manifold.h"

namespace Maniverse{

Constraint::Constraint(int total_size, std::vector<std::array<int, 3>> block_parameters, Function& func, std::vector<std::shared_ptr<Manifold>> manifolds){
	this->TotalSize = total_size;
	this->BlockParameters = block_parameters;
	this->Func = &func;
	this->Manifolds = manifolds;
	this->Lambda = 0;
	this->Gradient.resize(this->TotalSize); this->Gradient.setZero();
}	

void Constraint::setGradient(){
	for ( int jman = 0; jman < (int)this->Manifolds.size(); jman++ ){
		this->Manifolds[jman]->Ge = this->Func->Gradient[jman];
		this->Manifolds[jman]->getGradient();
		Eigen::VectorXd& cons_grad_i = this->Gradient;
		MV_SetBlock(cons_grad_i, jman, this->BlockParameters) = this->Manifolds[jman]->Gr;
	}
}

std::vector<Eigen::MatrixXd> Constraint::getGradient() const{
	std::vector<Eigen::MatrixXd> gs;
	MV_DecoupleBlock(this->Gradient, gs, this->BlockParameters);
	return gs;
}

Eigen::VectorXd Constraint::Hessian(Eigen::VectorXd Xvec) const{
	const int nmans = (int)this->Manifolds.size();
	std::vector<Eigen::MatrixXd> X(nmans);
	for ( int iman = 0; iman < nmans; iman++ ) X[iman] = MV_GetBlock(Xvec, iman, this->BlockParameters);

	std::vector<Eigen::MatrixXd> HeX = this->Func->Hessian(X);

	Eigen::VectorXd HrXvec = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iman = 0; iman < nmans; iman++ ){
		MV_SetBlock(HrXvec, iman, this->BlockParameters) = this->Manifolds[iman]->getHessian(HeX[iman], X[iman], 1);
	}
	return HrXvec;
}

void Iterate::setPoint(std::vector<Eigen::MatrixXd> ps, bool purify){
	if ( ps.size() != this->Manifolds.size() ) throw std::runtime_error("Wrong number of Points!");
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		this->Manifolds[iman]->setPoint(ps[iman], purify);
		MV_SetBlock(Point, iman, this->BlockParameters) = this->Manifolds[iman]->P;
	}
	for ( int icons = 0; icons < (int)this->Constraints.size(); icons++ ){
		for ( int jman = 0; jman < (int)this->Manifolds.size(); jman++ ){
			this->Constraints[icons].Manifolds[jman]->setPoint(ps[jman], purify);
		}
	}
}

std::vector<Eigen::MatrixXd> Iterate::getPoint() const{
	std::vector<Eigen::MatrixXd> ps(Manifolds.size());
	MV_DecoupleBlock(this->Point, ps, this->BlockParameters);
	return ps;
}

void Iterate::Calculate(std::vector<Eigen::MatrixXd> P, std::vector<int> derivatives){
	this->Objective->Calculate(P, derivatives);
	this->Value = this->Objective->Value;
	for ( Constraint& constraint : this->Constraints ){
		constraint.Func->Calculate(P, derivatives);
		this->Value += constraint.Lambda * constraint.Func->Value + 0.5 * this->Rho * constraint.Func->Value * constraint.Func->Value;
	}
}

void Iterate::setGradient(){
	this->setObjectiveGradient();
	this->Gradient = this->ObjectiveGradient;
	for ( Constraint& constraint : this->Constraints ){
		constraint.setGradient();
		this->Gradient += ( constraint.Lambda + this->Rho * constraint.Func->Value ) * constraint.Gradient;
	}
	for ( int icons = 0; icons < (int)this->Constraints.size(); icons++ ){
		Constraint& consi = this->Constraints[icons];
		consi.OrthogonalGradient = consi.Gradient;
		for ( int jcons = 0; jcons < icons; jcons++ ){
			Constraint& consj = this->Constraints[jcons];
			consi.OrthogonalGradient -= this->Inner(consi.Gradient, consj.OrthogonalGradient) * consj.OrthogonalGradient / this->Inner(consj.OrthogonalGradient, consj.OrthogonalGradient);
		}
	}
}

std::vector<Eigen::MatrixXd> Iterate::getGradient() const{
	std::vector<Eigen::MatrixXd> gs;
	MV_DecoupleBlock(this->Gradient, gs, this->BlockParameters);
	return gs;
}

Eigen::VectorXd Iterate::Hessian(Eigen::VectorXd Xvec) const{
	Eigen::VectorXd HrXvec = this->ObjectiveHessian(Xvec);
	for ( const Constraint& constraint : this->Constraints ){
		HrXvec += ( constraint.Lambda + this->Rho * constraint.Func->Value ) * constraint.Hessian(Xvec) + this->Rho * this->Inner(Xvec, constraint.Gradient) * constraint.Gradient;
	}
	return HrXvec;
}

void Iterate::setObjectiveGradient(){
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		this->Manifolds[iman]->Ge = this->Objective->Gradient[iman];
		this->Manifolds[iman]->getGradient();
		MV_SetBlock(ObjectiveGradient, iman, this->BlockParameters) = this->Manifolds[iman]->Gr;
	}
}

std::vector<Eigen::MatrixXd> Iterate::getObjectiveGradient() const{
	std::vector<Eigen::MatrixXd> gs;
	MV_DecoupleBlock(this->ObjectiveGradient, gs, this->BlockParameters);
	return gs;
}
Eigen::VectorXd Iterate::ObjectiveHessian(Eigen::VectorXd Xvec) const{
	const int nmans = (int)this->Manifolds.size();
	std::vector<Eigen::MatrixXd> X(nmans);
	for ( int iman = 0; iman < nmans; iman++ ) X[iman] = MV_GetBlock(Xvec, iman, this->BlockParameters);

	std::vector<Eigen::MatrixXd> HeX = this->Objective->Hessian(X);

	Eigen::VectorXd HrXvec = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iman = 0; iman < nmans; iman++ ){
		MV_SetBlock(HrXvec, iman, this->BlockParameters) = this->Manifolds[iman]->getHessian(HeX[iman], X[iman], 1);
	}
	return HrXvec;
}

std::vector<double> Iterate::calcLambda() const{
	const int ncons = this->Constraints.size();
	if ( ncons == 0 ) throw std::runtime_error("No constraints to calculate Lagrange multipliers for!");
	Eigen::VectorXd Gf = this->ObjectiveGradient;
	Eigen::MatrixXd Gg = Eigen::MatrixXd::Zero(Gf.size(), ncons);
	for ( int i = 0; i < ncons; i++ ) Gg.col(i) = this->Constraints[i].Gradient;
	const Eigen::VectorXd lambda = - Gg.colPivHouseholderQr().solve(Gf);
	return std::vector<double>(lambda.data(), lambda.data() + ncons);
}

void Iterate::setLambda(std::vector<double> lambda){
	const int ncons = this->Constraints.size();
	if ( ncons != (int)lambda.size() ) throw std::runtime_error("Wrong number of Lagrange multipliers!");
	for ( int i = 0; i < ncons; i++ ) this->Constraints[i].Lambda = lambda[i];
}

std::vector<double> Iterate::getLambda() const{
	std::vector<double> lambda;
	for ( const Constraint& constraint : this->Constraints ) lambda.push_back(constraint.Lambda);
	return lambda;
}

Eigen::VectorXd Iterate::ConstraintProjection(Eigen::VectorXd Xvec) const{
	for ( const Constraint& constraint : this->Constraints ){
		const Eigen::VectorXd& cons_grad = constraint.OrthogonalGradient;
		Xvec -= this->Inner(Xvec, cons_grad) * cons_grad / this->Inner(cons_grad, cons_grad);
	}
	return Xvec;
}

Iterate::Iterate(Function& objective, std::vector<std::shared_ptr<Manifold>> manifolds, std::vector<Function*> cons_funcs){
	this->Objective = &objective;

	const int nmans = (int)manifolds.size();
	this->Manifolds = manifolds;

	this->TotalSize = 0;
	for ( int iman = 0; iman < nmans; iman++ ){
		this->BlockParameters.push_back({
				this->TotalSize,
				(int)this->Manifolds[iman]->P.rows(),
				(int)this->Manifolds[iman]->P.cols()
		});
		this->TotalSize += this->Manifolds[iman]->P.size();
	}

	this->Point.resize(this->TotalSize); this->Point.setZero();
	this->Gradient.resize(this->TotalSize); this->Gradient.setZero();
	this->ObjectiveGradient.resize(this->TotalSize); this->ObjectiveGradient.setZero();
	for ( int iman = 0; iman < nmans; iman++ ){
		MV_SetBlock(Point, iman, this->BlockParameters) = Manifolds[iman]->P;
		MV_SetBlock(Gradient, iman, this->BlockParameters) = Manifolds[iman]->Gr;
		MV_SetBlock(ObjectiveGradient, iman, this->BlockParameters) = Manifolds[iman]->Gr;
	}

	for ( Function* cons_func : cons_funcs ){
		std::vector<std::shared_ptr<Manifold>> these_manifolds;
		for ( std::shared_ptr<Manifold> manifold : manifolds ){
			these_manifolds.push_back(manifold->Share());
		}
		this->Constraints.emplace_back(this->TotalSize, this->BlockParameters, *cons_func, these_manifolds);
	}
}

std::string Iterate::getName() const{
	std::string name = "";
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		if ( iman > 0 ) name += " * ";
		name += Manifolds[iman]->Name;
	}
	return name;
}

int Iterate::getDimension() const{
	int ndims = 0;
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ )
		 ndims += Manifolds[iman]->getDimension();
	return ndims;
}

double Iterate::Inner(Eigen::VectorXd X, Eigen::VectorXd Y) const{
	double inner = 0;
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		inner += this->Manifolds[iman]->Inner(MV_GetBlock(X, iman, this->BlockParameters), MV_GetBlock(Y, iman, this->BlockParameters));
	}
	return inner;
}

Eigen::VectorXd Iterate::Retract(Eigen::VectorXd X) const{
	Eigen::VectorXd Exp = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		MV_SetBlock(Exp, iman, this->BlockParameters) = this->Manifolds[iman]->Retract(MV_GetBlock(X, iman, this->BlockParameters));
	}
	return Exp;
}

Eigen::VectorXd Iterate::InverseRetract(Iterate& N) const{
	Eigen::MatrixXd Log = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		MV_SetBlock(Log, iman, this->BlockParameters) = this->Manifolds[iman]->InverseRetract(*(N.Manifolds[iman]));
	}
	return Log;
}

Eigen::VectorXd Iterate::TransportTangent(Eigen::VectorXd A, Eigen::VectorXd Y) const{
	Eigen::VectorXd B = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		MV_SetBlock(B, iman, this->BlockParameters) = this->Manifolds[iman]->TransportTangent(MV_GetBlock(A, iman, this->BlockParameters), MV_GetBlock(Y, iman, this->BlockParameters));
	}
	return B;
}

Eigen::VectorXd Iterate::TransportManifold(Eigen::VectorXd A, Iterate& N) const{
	Eigen::VectorXd B = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		MV_SetBlock(B, iman, this->BlockParameters) = this->Manifolds[iman]->TransportManifold(MV_GetBlock(A, iman, this->BlockParameters), *(N.Manifolds[iman]));
	}
	return B;
}

Eigen::VectorXd Iterate::TangentProjection(Eigen::VectorXd A) const{
	Eigen::VectorXd X = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		MV_SetBlock(X, iman, this->BlockParameters) = this->Manifolds[iman]->TangentProjection(MV_GetBlock(A, iman, this->BlockParameters));
	}
	return X;
}

Eigen::VectorXd Iterate::TangentPurification(Eigen::VectorXd A) const{
	Eigen::VectorXd X = Eigen::VectorXd::Zero(this->TotalSize);
	for ( int iman = 0; iman < (int)this->Manifolds.size(); iman++ ){
		MV_SetBlock(X, iman, this->BlockParameters) = this->Manifolds[iman]->TangentPurification(MV_GetBlock(A, iman, this->BlockParameters));
	}
	return X;
}

#ifdef __PYTHON__
void Init_Iterate(pybind11::module_& m){
	pybind11::classh<Constraint>(m, "Constraint")
		.def_readwrite("TotalSize", &Constraint::TotalSize)
		.def_readwrite("BlockParameters", &Constraint::BlockParameters)
		//.def_readwrite("Func", &Constraint::Func)
		.def_readwrite("Manifolds", &Constraint::Manifolds)
		.def(pybind11::init<int, std::vector<std::array<int, 3>>, Function&, std::vector<std::shared_ptr<Manifold>>>())
		.def_readwrite("Lambda", &Constraint::Lambda)
		.def_readwrite("Gradient", &Constraint::Gradient)
		.def_readwrite("OrthogonalGradient", &Constraint::OrthogonalGradient)
		.def("setGradient", &Constraint::setGradient)
		.def("getGradient", &Constraint::getGradient)
		.def("Hessian", &Constraint::Hessian);

	pybind11::classh<Iterate>(m, "Iterate")
		.def_readwrite("Point", &Iterate::Point)
		.def("setPoint", &Iterate::setPoint)
		.def("getPoint", &Iterate::getPoint)

		.def_readwrite("Value", &Iterate::Value)
		.def_readwrite("Manifolds", &Iterate::Manifolds)
		.def("Calculate", &Iterate::Calculate)
		.def_readwrite("Value", &Iterate::Value)
		.def_readwrite("Gradient", &Iterate::Gradient)
		.def("setGradient", &Iterate::setGradient)
		.def("getGradient", &Iterate::getGradient)
		.def("Hessian", &Iterate::Hessian)
		.def_readwrite("Preconditioner", &Iterate::Preconditioner)
		.def_readwrite("PreconditionerInv", &Iterate::PreconditionerInv)
		.def_readwrite("PreconditionerSqrt", &Iterate::PreconditionerSqrt)
		.def_readwrite("PreconditionerInvSqrt", &Iterate::PreconditionerInvSqrt)

		//.def_readwrite("Objective", &Iterate::Objective)
		.def_readwrite("ObjectiveGradient", &Iterate::ObjectiveGradient)
		.def("setObjectiveGradient", &Iterate::setObjectiveGradient)
		.def("getObjectiveGradient", &Iterate::getObjectiveGradient)
		.def("ObjectiveHessian", &Iterate::ObjectiveHessian)

		.def_readwrite("Constraints", &Iterate::Constraints)
		.def("calcLambda", &Iterate::calcLambda)
		.def("setLambda", &Iterate::setLambda)
		.def("getLambda", &Iterate::getLambda)
		.def_readwrite("Rho", &Iterate::Rho)
		.def("ConstraintProjection", &Iterate::ConstraintProjection)

		.def_readwrite("TotalSize", &Iterate::TotalSize)
		.def_readwrite("BlockParameters", &Iterate::BlockParameters)

		.def(pybind11::init<Function&, std::vector<std::shared_ptr<Manifold>>, std::vector<Function*>>(), pybind11::arg("objective"), pybind11::arg("manifolds"), pybind11::arg("cons_funcs") = std::vector<Function*>())

		.def("getName", &Iterate::getName)
		.def("getDimension", &Iterate::getDimension)

		.def("Inner", &Iterate::Inner)
		.def("Retract", &Iterate::Retract)
		.def("InverseRetract", &Iterate::InverseRetract)
		.def("TransportTangent", &Iterate::TransportTangent)
		.def("TransportManifold", &Iterate::TransportManifold)

		.def("TangentProjection", &Iterate::TangentProjection)
		.def("TangentPurification", &Iterate::TangentPurification);
}
#endif

}
