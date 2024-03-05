using LinearAlgebra

# Generate a 10x10 non-positive definite matrix Q
Q = rand(10, 10)
Q = Q * Q' - 5I

# Generate a 10-dimensional vector q
q = rand(10)

# Objective function
objective(x) = 0.5 * x' * Q * x + q' * x

# Gradient function
gradient(x) = Q * x + q

# Sigmoid function to map values into (0, 1)
sigmoid(x) = 1.0 / (1.0 + exp(-x))  

# projection

# interior method
function trust_region_method_interior(obj, grad, x0; tol=1e-6, max_iter=1000, μ=1e-3)
    x = x0
    Δ = 1.0  # Initial trust region radius
    buffer = 1e-10  # Buffer to avoid log(0) or log(1)
    for i in 1:max_iter
        # Modify the gradient to include the penalty for boundary violation
        x = max.(min.(x, 1 - buffer), buffer)  # Ensure x stays within (0+buffer, 1-buffer)
        penalty_grad = μ * (-1 ./ x - 1 ./ (1 .- x))
        g = grad(x) + penalty_grad
        p = -Δ * normalize(g)
        x_new = x + p
        # Apply the buffer again to ensure no boundary violations
        x_new = max.(min.(x_new, 1 - buffer), buffer)
        penalty = -μ * sum(log.(x_new) + log.(1 .- x_new))
        if obj(x_new) + penalty < obj(x) + penalty
            x = x_new
        end
        Δ *= norm(g) > tol ? 0.8 : 1.2
        if norm(g) < tol
            break
        end
    end
    return x
end

# Initialize a random point within box [0, 1] as the starting point
x0 = rand(10)

x_opt_interior = trust_region_method_interior(objective, gradient, x0)

println("Optimal solution: ", x_opt_interior)
println("Minimum value: ", objective(x_opt_interior))
