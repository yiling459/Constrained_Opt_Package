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

# An implementation of trust region method
function trust_region_method(obj, grad, x0; tol=1e-6, max_iter=1000)
    x = x0
    Δ = 1.0  # Initial trust region radius
    for i in 1:max_iter
        g = grad(x)
        # The solution to the trust region subproblem can be obtained by solving a quadratic program, here simplified to a step in the gradient direction
        p = -Δ * normalize(g)
        # Simplified acceptance criterion: if the objective function value decreases, accept the step.
        if obj(x + p) < obj(x)
            x += p
        end
        # Update the trust region radius
        if norm(g) > tol
            Δ *= 0.8 
        else
            Δ *= 1.2
        end
        # Check for convergence
        if norm(g) < tol
            break
        end
    end
    return x
end

# Initialize a random point within box [0, 1] as the starting point
x0 = rand(10)

# Apply the trust region method
x_opt = trust_region_method(objective, gradient, x0)

println("Optimal solution: ", x_opt)
println("Minimum value: ", objective(x_opt))
