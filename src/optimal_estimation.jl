
@doc raw"""
    solve_(prop:OptimalEstimation)

Rodgers-style linear optimal estimation, solves
```math
\begin{pmatrix} S_\epsilon^{-\frac{1}{2}}K \\ S_a^{-\frac{1}{2}} \end{pmatrix} x = \begin{pmatrix} S_\epsilon^{-\frac{1}{2}}y \\ 0 \end{pmatrix}
```
"""
function solve_stable(prob::OptimalEstimationProblem{FT}; errorAnalysis::Bool=true) where {FT}
    @unpack K, Sₑ, Sₐ, δy = prob
    n,m = size(K)
    # Compute S^{-0.5} using cholesky decomposition
    sqrtSₑ⁻¹ = inv(cholesky!(Sₑ).L);
    sqrtSₐ⁻¹ = inv(cholesky!(Sₐ).L);
    
    # A bit tediuous that way but reduces memory allocation:
    lhs = Array{FT}(undef,n+m,m)
    rhs = Array{FT}(undef,n+m)
    # K̃ = [sqrtSₑ⁻¹ * K; sqrtSₐ⁻¹]; ỹ = [sqrtSₑ⁻¹ * δy; 0]
    @views lhs[1:n,1:m]     = sqrtSₑ⁻¹ * K
    @views lhs[n+1:n+m,1:m] = sqrtSₐ⁻¹
    @views rhs[1:n]         = sqrtSₑ⁻¹ * δy
    @views rhs[n+1:end]    .= 0
    
    Kqr = qr!(lhs)
    
    
    # Solve like in Rodgers using QR decomposition, eq. 5.43, 
    Δx̂ = Kqr \ rhs
    
    # Perform error analysis:
    if errorAnalysis
        Sₑ⁻¹ = sqrtSₑ⁻¹' * sqrtSₑ⁻¹
        # Compute Ŝ from K̃ = [sqrtSₑ⁻¹ * K; sqrtSₐ⁻¹]
        L = LinearAlgebra.inv!(UpperTriangular(Kqr.R));#UpperTriangular(Kqr.R)\I
        Ŝ = L * L'
        A = Ŝ * K' * Sₑ⁻¹ * K
        # residual r and reduced Χᵣ²
        r  = K*Δx̂ - δy
        # Reduced χ²
        χ² =  (r'* Sₑ⁻¹ *r)/(n-m)
        return OEsolution(Δx̂,Ŝ,A,r,χ²)
    end
    # Add xₐ here later?
    return Δx̂
end

@doc raw"""
    solve(prop:OptimalEstimation)

Rodgers-style linear optimal estimation, solves
\hat{x} = x_a + (K^TS_\epsilon^{-1} K + S_a^{-1})^{-1} K^T S_\epsilon^{-1}\delta y
"""
function solve(prob::OptimalEstimationProblem; errorAnalysis::Bool=true)
    @unpack K, Sₑ, Sₐ, δy = prob
    n,m = size(K)
    # Inverse(s)
    Sₑ⁻¹ = inv(Sₑ)
    Sₐ⁻¹ = inv(Sₐ)
    # Will be re-used:
    KᵀSₑ⁻¹K = K' * Sₑ⁻¹ * K
    
    # Posterior covariance matrix:
    Ŝ = inv(KᵀSₑ⁻¹K + Sₐ⁻¹)
    
    # Estimated state vector (eq 4.5):
    Δx̂ = Ŝ * K'Sₑ⁻¹ * δy
    if errorAnalysis
        # Averaging kernel matrix:
        A = Ŝ * KᵀSₑ⁻¹K
        # residual r and reduced Χᵣ²
        r  = K*Δx̂ - δy
        # Reduced χ²
        χ² =  (r'*Sₑ⁻¹*r)/(n-m)
        return OEsolution(Δx̂,Ŝ,A,r,χ²)
    end
    return Δx̂
end

# sqrtSₑ⁻¹ = sqrt.(inv(Sₑ));
# Solve like in Rodgers, eq. 5.43, supposedly more stable
#    lh = [sqrtSₑ⁻¹ * K; Sₐ]
#    rh = [sqrtSₑ⁻¹ * δy; zeros(nx)]
#    return lh \ rh