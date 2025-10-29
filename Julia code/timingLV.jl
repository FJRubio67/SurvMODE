using DifferentialEquations
using BenchmarkTools

# Lotka–Volterra system
function lotka_volterra!(du, u, p, t)
    α, β, δ, γ = p
    du[1] = α*u[1] - β*u[1]*u[2]     # Prey
    du[2] = δ*u[1]*u[2] - γ*u[2]     # Predator
end

# Parameters and initial conditions
α, β, δ, γ = 1.5, 1.0, 0.75, 1.0
u0 = [10.0, 5.0]
tspan = (0.0, 20.0)
p = (α, β, δ, γ)

# Define the ODE problem
prob = ODEProblem(lotka_volterra!, u0, tspan, p)

# Warm-up run (important for JIT compilation)
@time solve(prob, Tsit5())

# Benchmark timing (more accurate)
@btime sol = solve($prob, Tsit5())
