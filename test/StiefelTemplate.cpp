#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include <Maniverse/Manifold/GeneralizedStiefel.h>
#include <Maniverse/Manifold/Stiefel.h>
#include <Maniverse/Manifold/StiefelTemplate.h>

namespace mv = Maniverse;

void require(bool condition, const char* message){
	if ( ! condition ) throw std::runtime_error(message);
}

void requireApprox(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected, const char* message){
	require(actual.isApprox(expected, 1e-10), message);
}

template<typename OldManifold, typename NewManifold>
void comparePolarOperations(OldManifold& old, NewManifold& modern, const Eigen::MatrixXd& ambient){
	require(old.getDimension() == modern.getDimension(), "Dimension mismatch");
	require(std::abs(old.Inner(ambient, ambient) - modern.Inner(ambient, ambient)) < 1e-10, "Inner product mismatch");
	requireApprox(old.Flat(ambient), modern.Flat(ambient), "Flat mismatch");
	requireApprox(old.Sharp(ambient), modern.Sharp(ambient), "Sharp mismatch");
	requireApprox(old.TangentProjection(ambient), modern.TangentProjection(ambient), "Tangent projection mismatch");
	requireApprox(old.TangentPurification(ambient), modern.TangentPurification(ambient), "Tangent purification mismatch");

	const Eigen::MatrixXd oldTangent = old.TangentProjection(ambient);
	const Eigen::MatrixXd newTangent = modern.TangentProjection(ambient);
	requireApprox(old.Retract(oldTangent), modern.Retract(newTangent), "Retraction mismatch");

	OldManifold oldDestination = old;
	NewManifold newDestination = modern;
	oldDestination.setPoint(old.Retract(oldTangent), false);
	newDestination.setPoint(modern.Retract(newTangent), false);
	requireApprox(old.InverseRetract(oldDestination), modern.InverseRetract(newDestination), "Inverse retraction mismatch");
	requireApprox(
		old.TransportTangent(ambient, oldTangent),
		modern.TransportTangent(ambient, newTangent),
		"Tangent transport mismatch"
	);
	requireApprox(
		old.TransportManifold(ambient, oldDestination),
		modern.TransportManifold(ambient, newDestination),
		"Manifold transport mismatch"
	);

	old.Ge = ambient;
	modern.Ge = ambient;
	old.getGradient();
	modern.getGradient();
	requireApprox(old.Gr, modern.Gr, "Gradient mismatch");
	requireApprox(
		old.getHessian(ambient, oldTangent),
		modern.getHessian(ambient, newTangent),
		"Hessian mismatch"
	);

	std::unique_ptr<mv::Manifold> oldClone = old.Clone();
	std::unique_ptr<mv::Manifold> newClone = modern.Clone();
	require(typeid(*oldClone) == typeid(old), "Legacy clone changed dynamic type");
	require(typeid(*newClone) == typeid(modern), "Template clone changed dynamic type");
	require(typeid(*modern.Share()) == typeid(modern), "Template share changed dynamic type");

	OldManifold oldPurified = old;
	NewManifold newPurified = modern;
	oldPurified.setPoint(ambient, true);
	newPurified.setPoint(ambient, true);
	requireApprox(oldPurified.P, newPurified.P, "Point purification mismatch");
}

void checkStiefel(){
	const Eigen::MatrixXd p = Eigen::MatrixXd::Identity(5, 3);
	const Eigen::MatrixXd ambient = Eigen::MatrixXd::Random(5, 3);
	mv::Stiefel legacy(p);
	mv::StiefelFromTemplate templated(p);
	comparePolarOperations(legacy, templated, ambient);

	for (const std::string geodesic : {"EXACT", "QR"}){
		mv::Stiefel oldWithGeodesic(p, geodesic);
		mv::StiefelFromTemplate newWithGeodesic(p, geodesic);
		requireApprox(
			oldWithGeodesic.Retract(ambient),
			newWithGeodesic.Retract(ambient),
			"Stiefel retraction differs for selected geodesic"
		);
		if ( geodesic == "QR" ){
			requireApprox(
				oldWithGeodesic.TransportTangent(ambient, ambient),
				newWithGeodesic.TransportTangent(ambient, ambient),
				"Stiefel QR transport mismatch"
			);
		}
	}
}

void checkGeneralizedStiefel(){
	Eigen::MatrixXd B = Eigen::MatrixXd::Zero(5, 5);
	B.diagonal() << 1.0, 2.0, 3.0, 4.0, 5.0;
	Eigen::MatrixXd p = Eigen::MatrixXd::Zero(5, 3);
	p.topRows(3) = B.diagonal().head(3).cwiseSqrt().cwiseInverse().asDiagonal();
	const Eigen::MatrixXd ambient = Eigen::MatrixXd::Random(5, 3);
	mv::GeneralizedStiefel legacy(p, B);
	mv::GeneralizedStiefelFromTemplate templated(p, B);
	comparePolarOperations(legacy, templated, ambient);

	mv::GeneralizedStiefel oldWithGeodesic(p, B, "EXACT");
	mv::GeneralizedStiefelFromTemplate newWithGeodesic(p, B, "EXACT");
	requireApprox(
		oldWithGeodesic.Retract(ambient),
		newWithGeodesic.Retract(ambient),
		"Generalized Stiefel exact retraction mismatch"
	);
}

int main(){
	try{
		checkStiefel();
		checkGeneralizedStiefel();
		std::cout << "StiefelTemplate Success!" << std::endl;
	}catch (const std::exception& error){
		std::cout << "StiefelTemplate Failed: " << error.what() << std::endl;
		return 1;
	}
}
