" Solve a simple Linear least squares problem"
function solve(prob::LinearLeastSquares)
    @unpack K, y = prob
    K\y
end

"Solve a weighted linear least squares problem"
function solve(prob::WeightedLinearLeastSquares)
    @unpack K, y, Sₑ = prob
    Sₑ⁻¹ = inv(Sₑ)
    # Will be re-used:
    KᵀSₑ⁻¹K = K' * Sₑ⁻¹ * K
    # Posterior covariance matrix:
    Ŝ = inv(KᵀSₑ⁻¹K)
    # Averaging kernel matrix:
    A = Ŝ * KᵀSₑ⁻¹K
    # State vector:
    x̂ = Ŝ * K'Sₑ⁻¹ * y     
    #sqrtSₑ⁻¹ = sqrt(inv(Sₑ));
    #Ŝ = inv(KᵀSₑ⁻¹K 
    #(sqrtSₑ⁻¹ * K) \ (sqrtSₑ⁻¹ * y)
end