using LinearAlgebra
using BenchmarkTools
ny = 1000
nx = 100
Sₐ = (Diagonal(abs.(rand(nx))));
Sₑ = (Diagonal((3rand(ny))));
#Sₐ = (abs.(rand(nx,nx)));
#Sₑ = ((10*rand(ny,ny)));
K = rand(ny,nx);
y = randn(ny);

prob = OptimalEstimation.OptimalEstimationProblem(K, Sₑ, y, Sₐ)


#OptimalEstimation.solve(prob) ≈ OptimalEstimation.solve_stable(prob)
@time OptimalEstimation.solve(prob, errorAnalysis=true) ;
@time OptimalEstimation.solve_stable(prob,errorAnalysis=true);