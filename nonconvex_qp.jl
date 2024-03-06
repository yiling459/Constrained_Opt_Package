using LinearAlgebra
using Plots

# Generate a 10x10 non-positive definite matrix Q
Q = rand(10, 10)
Q = Q * Q' - 5I

# Generate a 10-dimensional vector q
q = rand(10)

# Objective function
objective(x) = 0.5 * x' * Q * x + q' * x

# Gradient function
gradient(x) = Q * x + q

function trust_region_method_interior(obj, grad, x0; tol=1e-6, max_iter=1000, μ=1e-3)
    x = x0
    Δ = 1.0  # Initial trust region radius
    obj_values = []
    penalties = []
    for i in 1:max_iter
        x = max.(min.(x, 1 - 1e-10), 1e-10)
        penalty_grad = μ * (-1 ./ x - 1 ./ (1 .- x))
        g = grad(x) + penalty_grad
        p = -Δ * normalize(g)
        x_new = x + p
        x_new = max.(min.(x_new, 1 - 1e-10), 1e-10)
        penalty = -μ * sum(log.(x_new) + log.(1 .- x_new))
        push!(obj_values, obj(x_new) + penalty)
        push!(penalties, penalty)
        if obj(x_new) + penalty < obj(x) + penalty
            x = x_new
        end
        Δ *= norm(g) > tol ? 0.8 : 1.2
        if norm(g) < tol
            break
        end
    end
    return x, obj_values, penalties
end

x0 = rand(10)
x_opt, obj_values, penalties = trust_region_method_interior(objective, gradient, x0)

# Plotting the objective function values and penalties
p1 = plot(obj_values, title = "Objective Function Value", xlabel = "Iteration", ylabel = "Value")
p2 = plot(penalties, title = "Penalty", xlabel = "Iteration", ylabel = "Penalty Value")
plot(p1, p2, layout = (2, 1), legend = false)
