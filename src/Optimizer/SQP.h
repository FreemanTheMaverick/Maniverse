#include "../Manifold/Manifold.h"
#include "../LinearSolver/LinearSolver.h"

namespace Maniverse{

void initLinearSolverForNormal(LinearSolver& ls, Iterate& M);

void initLinearSolverForProjectedCG(LinearSolver& ls, Iterate& M);

}