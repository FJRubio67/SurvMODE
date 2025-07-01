#=
****************************************************************************
Required packages
****************************************************************************
=#

using Plots
using DifferentialEquations
using LinearAlgebra
using CSV
using LSODA
using Optim
using Distributions
using Random
using AdaptiveMCMC
using Tables
using DelimitedFiles
using Statistics
using Survival
using DataFrames
using FreqTables
using Sundials
using ForwardDiff
using Turing
using StatsPlots
using StatsFuns
using JLD2

#=
****************************************************************************
# Additional routines 
****************************************************************************
=#

include("routines.jl")


#=
****************************************************************************
Data preparation
****************************************************************************
=#


#= Data =#
df_full = CSV.File("rotterdamFull.csv");
df0 = CSV.File("rotterdam.csv");


#= Data of interest =#
size_bin = collect(df_full.size)
sizes = zeros(size(size_bin)[1])

for i in 1:size(size_bin)[1]
    if size_bin[i] == ">50"
        sizes[i] = 1.0
    else
        sizes[i] = 0.0
    end
  end


df = DataFrame( time = collect(df_full.rtime)./365.25, 
                status = collect(df_full.recur),
                nodes = scale(collect(df_full.nodes)),
                agec = scale(collect(df_full.age)),
                sizes = sizes,
                trt = convert(Vector{Float64}, collect(df_full.chemo)))

# Removing inconsistent Data
indcon = zeros(size(df_full)[1])
for i in 1:size(df_full)[1]
    if ((df_full.rtime[i] < df_full.dtime[i]) && (df_full.recur[i] == 0) && (df_full.death[i] == 1))
        indcon[i] = 0
    else
        indcon[i] = 1
    end
end

# Inconsistent data index
indcon = collect(Bool, (indcon))

df = df[indcon,:]

# Sorting df by time order
sorted_indices = sortperm(df[:, :time])

df = df[sorted_indices,:]

# Sample size
n = size(df)[1]

#= Vital status =#
status = collect(Bool, (df.status));

#= Survival times =#
times = df.time;

# Time grids
tspan0 = hcat(zeros(n), df.time);

tspan00 = vcat(0.0, df.time);
tmax = maximum(df.time)

# Initial conditions (h,q,H)
u0 = [1.0e-2, 1.0e-6, 0.0]

# Initial conditions (log h,log q,H)
lu0 = [log(1.0e-2), log(1.0e-6), 0.0]

#=
****************************************************************************
Fitting the model without covariates: MLE and MAP
****************************************************************************
=#

# MLE: No covariates
optmle0 = optimize(mlog_likL0, [0.0,0.0,0.0,0.0], method=NelderMead(), iterations=10000)

MLE0 = optmle0.minimizer


# MAP: No covariates
optmap0 = optimize(mlog_postL0, [0.0,0.0,0.0,0.0], method=NelderMead(), iterations=10000)

MAP0 = optmap0.minimizer

#=
****************************************************************************
Model building: variable screening/selection
****************************************************************************
=#

# Design matrix including variables of interest for model building
# age + sizes + nodes + trt
des_full = hcat( df.agec, df.sizes, df.nodes, df.trt)

# Hyper-parameters for g-priors
gs = [1.0,1.0,1.0,0.01].*(n-0.5*sum(status))

# Number of iterations for optimisation
M = 100000

# Index for saturated model
init_all = chain0 =  vcat([1,1,1,1,1]',[1,1,1,1,1]',[1,1,1,1,1]',[1,1,1,1,1]')

# MAP under g-prior
OPT = HRMAP(des_full, init_all, gs, M, zeros(length(init_all)))

# Index for model without covariates as initial model
chain0 =  vcat([1,0,0,0,0]',[1,0,0,0,0]',[1,0,0,0,0]',[1,0,0,0,0]')

# Seed, number of Gibbs iterations, burn-in and number of iterations in optimiser
seed = 123
niter = 1100
init_ind = copy(chain0)
burnin = 100
M = 10000

#=
------------------------------------------------
Variable selection step
Uncomment to run (requires ~36 hours)
------------------------------------------------
=#

#=
# Running Gibbs sampler for variable selection
modsel = Gibbs_varsel(seed, niter, init_ind, des_full, gs, burnin, M)

# Posterior Inclusion Probabilities (intercept,age,sizes,nodes,trt)
modsel.pip 

# Model probabilities
modsel.modpp

# Highest posterior probability model in matrix form
key2matrix("1_1_1_1_0_0_0_1_0_0_0_1_1_1_1_0_1_0_1_1")

# Median model in matrix form (same as MAP)
key2matrix("1_1_1_1_0_0_0_1_0_0_0_1_1_1_1_0_1_0_1_1")

# Save to file
@save "modsel.jld2" modsel

# Load from file
#@load "modsel.jld2"
=#

# Output
#=
julia> modsel.modpp
OrderedDict{String, Float64} with 75 entries:
  "1_1_1_1_0_0_0_1_0_0_0_1_1_1_1_0_1_0_1_1" => 0.684685
  "1_1_1_1_0_0_0_1_0_0_0_1_1_1_1_0_1_0_1_0" => 0.0510511
  "1_1_1_1_0_0_0_1_0_0_0_1_1_1_1_0_1_1_0_1" => 0.028028
  "1_1_1_1_0_0_0_1_0_0_0_1_1_1_1_0_1_1_1_1" => 0.027027
  "1_1_1_1_0_0_0_1_0_0_0_1_1_1_1_1_1_1_1_1" => 0.023023
  "1_1_1_1_0_0_0_1_1_0_1_0_1_1_1_0_1_0_1_1" => 0.015015
  "1_1_1_1_0_0_0_1_0_1_1_0_1_0_1_1_1_0_1_0" => 0.014014
  "1_1_1_1_0_0_0_1_1_1_0_0_1_1_1_0_1_0_1_1" => 0.01001
  "1_1_1_1_0_0_0_1_0_0_0_1_1_0_1_1_1_0_1_0" => 0.00800801
  "1_1_1_1_0_0_0_1_0_0_0_1_1_1_1_1_1_1_0_1" => 0.00800801
=#


#=
****************************************************************************
Data preparation for best model: initial checks
****************************************************************************
=#

# Design matrices for the regression models for each parameter 
#(intercept,age,sizes,nodes,trt)
# lambda: intercept + nodes + trt
des_l = hcat(ones(n), des_full[:,3:4]);
p_l = size(des_l)[2];
# kappa: intercept + nodes
des_k = hcat(ones(n), des_full[:,3]);
p_k = size(des_k)[2];
# alpha: intercept + nodes + trt
des_a = hcat(ones(n), des_full[:,3:4]);
p_a = size(des_a)[2];
# beta: intercept + age + sizes + trt
des_b = hcat(ones(n), des_full[:,[1,2,4]]);
p_b = size(des_b)[2];

# Intercept positions
indint = [1, p_l + 1, p_l + p_k + 1, p_l + p_k + p_a + 1]
indbeta = deleteat!(collect(1:(p_l+p_k+p_a+p_b)), indint)



#= 
**********************************************************************************
MLE
**********************************************************************************
=#

# Index for best model (MAP)
index_best = key2matrix("1_1_1_1_0_0_0_1_0_0_0_1_1_1_1_0_1_0_1_1")

# Initial point for MLE based on MAP for saturated model
initmle =  OPT[1].minimizer[findall(reshape(index_best', :) .> 0)]

# Optimisation step to find the MLE
OPT_MLER = HRMLE(des_full, index_best, M, initmle)

# MLE for best regression model
MLER = vec(OPT_MLER[1].minimizer)

# Save MLER
writedlm("MLER.txt", MLER)


#= 
**********************************************************************************
MAP for best regression model based on weakly informative priors
Only for initial guess for MCMC Sampling
**********************************************************************************
=#

# Optimisation step to find the MAP
OPT_MAPR = HRMAPW(des_full, times, status, index_best, M, OPT_MLER[1].minimizer)

MAPR = vec(OPT_MAPR[1].minimizer)

# Save MAP for best regression model
writedlm("MAPR.txt", MAPR)


#=
****************************************************************************
Posterior sampling using AdaptiveMCMC.jl
https://docs.juliahub.com/AdaptiveMCMC
****************************************************************************
=#

# Number of iterations
NMC = 150000
Random.seed!(123)
#mcmc_best = adaptive_rwm(MAPR, log_postLMTW, NMC; algorithm=:ram, b = 0)
#mcmc_best = adaptive_rwm(MAPR, log_postLMTW, NMC; algorithm=:aswam, b = 0)
mcmc_best = adaptive_rwm(MAPR, log_postLMTW, NMC; algorithm=:am, b = 0)

# Burn-in and thinning
burn = 50000
thin = 100

# Visualising results
plot_histograms_by_column(transpose(mcmc_best.X[1:size(MLER)[1], burn:thin:end]))

plot_traceplots_by_column(transpose(mcmc_best.X[1:size(MLER)[1], burn:thin:end]))

# Save posterior samples
postsamp_mcmc = Tables.table(transpose(mcmc_best.X[:, burn:thin:end]))

CSV.write("postsamp_mcmc.csv", postsamp_mcmc)


#=
------------------------------------------------------------------------
Optimisation using the output from the MCMC sampler
------------------------------------------------------------------------
=#
post_median = vec(mapslices(median, transpose(mcmc_best.X[:, burn:thin:end]); dims=1))

# Save posterior median for best regression model
writedlm("MEDR.txt", post_median)

# MLE
OPT_MLER2 = HRMLE(des_full, index_best, M, post_median)

MLER2 = vec(OPT_MLER2[1].minimizer)

AICR = 2*OPT_MLER[1].minimum + 2*length(MLER)

BICR = 2*OPT_MLER[1].minimum + log(n)*length(MLER)

# Save improved MLE for best regression model
writedlm("MLER2.txt", MLER2)

# Optimisation step to find the MAP
OPT_MAPR2 = HRMAPW(des_full, times, status, index_best, M, post_median)

MAPR2 = vec(OPT_MAPR2[1].minimizer)

# Save MAP for best regression model
writedlm("MAPR2.txt", MAPR2)

# Compare MLER, post_median, and MAPR
hcat(MLER, MLER2, post_median, MAPR, MAPR2)

#= 
**********************************************************************************
Sampling from the posterior of selected regression model using 
an asymptotic normal approximation
**********************************************************************************
=#

# Hessian of negative log-posterior 
HESSB = Hermitian(ForwardDiff.hessian(mlog_postLMTW, MAPR2))

# Covariance matrix for normal approximation
SigmaB = Hermitian(inv(HESSB))

# Checking that SigmaB is Hermitian
ishermitian(SigmaB)

writedlm("SigmaB.txt", SigmaB)

# Create a MultivariateNormal distribution
distB = MultivariateNormal(MAPR2, SigmaB)

# Generate posterior samples using a normal approximation
Random.seed!(1234)
n_samples = 10000

postsamp_app = transpose(rand(distB, n_samples))

# Visualising results
plot_histograms_by_column(postsamp_app)

plot_traceplots_by_column(postsamp_app)

# Save posterior samples
writedlm("app_post_samples.txt", postsamp_app)

#=
**********************************************************************************
Comparisons
**********************************************************************************
=#

# Point estimates
hcat(MLER, vec(mean(postsamp_app,dims=1)), vec(mean(mcmc_best.X[1:size(MLER)[1], burn:thin:end],dims=2)), MAPR)

# Comparison MCMC samples and normal approximation: interval estimation
hcat(vec(mapslices(x -> "$(mean(x)) ± $(1.96std(x))", mcmc_best.X[1:size(MLER)[1], burn:thin:end], dims=2)),
vec(mapslices(x -> "$(mean(x)) ± $(1.96std(x))", postsamp_app, dims=1)))

