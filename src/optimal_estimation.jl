
@doc raw"""
    solve_(prop:OptimalEstimation)

Rodgers-style linear optimal estimation, solves
```math
\begin{pmatrix} S_\epsilon^{-\frac{1}{2}}K \\ S_a^{-\frac{1}{2}} \end{pmatrix} x = \begin{pmatrix} S_\epsilon^{-\frac{1}{2}}y \\ 0 \end{pmatrix}
```
"""
function solve_stable(prop::OptimalEstimationProblem)
    @unpack K, Sₑ, Sₐ, δy = prob
    # Maybe check if this is a vector or not?
    sqrtSₑ⁻¹ = sqrt.(inv(Sₑ));
    # Solve like in Rodgers, eq. 5.43, supposedly more stable
    lh = [sqrtSₑ⁻¹ * K; Sₐ]
    rh = [sqrtSₑ⁻¹ * δy; zeros(nx)]
    return lh \ rh
end

@doc raw"""
    solve(prop:OptimalEstimation)

Rodgers-style linear optimal estimation, solves
\hat{x} = x_a + (K^TS_\epsilon^{-1} K + S_a^{-1})^{-1} K^T S_\epsilon^{-1}\delta y
"""
function solve(prop::OptimalEstimationProblem)
    @unpack K, Sₑ, Sₐ, δy = prob
    # Inverse(s)
    Sₑ⁻¹ = inv(Sₑ)
    Sₐ⁻¹ = inv(Sₐ)
    # Will be re-used:
    KᵀSₑ⁻¹K = K' * Sₑ⁻¹ * K
    # Posterior covariance matrix:
    Ŝ = inv(KᵀSₑ⁻¹K + Sₐ⁻¹)
    # Averaging kernel matrix:
    A = Ŝ * KᵀSₑ⁻¹K
    # Estimated state vector (eq 4.5):
    x̂ = xₐ + Ŝ * K'Sₑ⁻¹ * δy
end

# sqrtSₑ⁻¹ = sqrt.(inv(Sₑ));
# Solve like in Rodgers, eq. 5.43, supposedly more stable
#    lh = [sqrtSₑ⁻¹ * K; Sₐ]
#    rh = [sqrtSₑ⁻¹ * δy; zeros(nx)]
#    return lh \ rh