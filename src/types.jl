abstract type AbstractInverseModel end
abstract type AbstractInverseSolution end

Base.@kwdef struct LinearLeastSquares{FT} <: AbstractInverseModel
    "Jacobian Matrix K"
    K::Matrix{FT}
    "Measurement vector y"
    y::Vector{FT}
end

Base.@kwdef struct WeightedLinearLeastSquares{FT,FT2<:AbstractArray{FT}}  <: AbstractInverseModel
    "Jacobian Matrix K"
    K::Matrix{FT}
    "Measurement covariance matrix Sₑ"
    Sₑ::FT2
    "Measurement vector y"  
    y::Vector{FT}
end

Base.@kwdef struct OptimalEstimationProblem{FT,FT2<:AbstractArray{FT},FT3<:AbstractArray{FT}} <: AbstractInverseModel
    "Jacobian Matrix K"
    K::Matrix{FT}
    "Measurement covariance matrix Sₑ"
    Sₑ::FT2
    "Measurement vector δy (y-Kxₐ) in linear case, (y-F(xᵢ)+Kᵢ(xᵢ-xₐ)) in non-linear case"
    δy::Vector{FT}
    "Prior state covariance matrix Sₐ"
    Sₐ::FT3
end

Base.@kwdef struct OEsolution{FT} <: AbstractInverseSolution
    "Retrieved state vector x̂"
    x̂::Vector{FT}
    "Posterior Covariance Matrix Ŝ"
    Ŝ::Matrix{FT}
    "Averaging kernel Matrix A"
    A::Matrix{FT}
    "residual vector r"
    r::Vector{FT}
    "reduced χ²"
    χ²::FT
    
end