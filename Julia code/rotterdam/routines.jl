#=
****************************************************************************
****************************************************************************
Additional functions
****************************************************************************
****************************************************************************
=#

# Standardisation function
scale(x) = (x .- mean(x)) ./ std(x)

function key2matrix(key)
    out = reshape(parse.(Int, split(key, "_")), (4, 5))
    return out
end

# Ratio of exponentials for large values of the argument
# exp(x)/[ exp(x) + exp(y) ]
function exp_ratio(x, y)
    max_log = max(x, y)
    exp_shifted_x = exp(x - max_log)
    exp_shifted_y = exp(y - max_log)
    return exp_shifted_x / (exp_shifted_x + exp_shifted_y)
end

# Weighted ratio of exponentials for large values of the argument
# wx*exp(x)/[ wx*exp(x) + wy*exp(y) ]
function wexp_ratio(x, y, wx, wy)
    max_log = max(x, y)
    exp_shifted_x = exp(x - max_log)
    exp_shifted_y = exp(y - max_log)
    return exp_shifted_x*wx / (exp_shifted_x*wx + exp_shifted_y*wy)
end



# Function to plot histograms of the columns of a matrix
function plot_histograms_by_column(data)
    num_cols = size(data, 2)
    plots = []

    for i in 1:num_cols
        hist = histogram(data[:, i], title="Parameter $i", legend=false)
        push!(plots, hist)
    end

    plot(plots..., layout=(num_cols, 1), size=(800, 300 * num_cols))
end


# Function to create traceplots of the columns of a matrix
function plot_traceplots_by_column(data)
    num_cols = size(data, 2)
    plots = []

    for i in 1:num_cols
        hist = plot(data[:, i], title="Parameter $i", legend=false)
        push!(plots, hist)
    end

    plot(plots..., layout=(num_cols, 1), size=(800, 300 * num_cols))
end

#=
****************************************************************************
****************************************************************************
Hazard-Response model
****************************************************************************
****************************************************************************
=#

# Hazard-Response ODE model
function HazResp(dh, h, p, t)
    # Model parameters
    lambda, kappa, alpha, beta = p

    # ODE System
    dh[1] = lambda * h[1] * (1 - h[1] / kappa) - alpha * h[1] * h[2] # hazard
    dh[2] = beta * h[2] * (1 - h[2] / kappa) - alpha * h[1] * h[2] # response
    dh[3] = h[1] # cumulative hazard
    return nothing
end

# Jacobian for Hazard-Response model

function jacHR(J, u, p, t)
    # Parameters
    lambda, kappa, alpha, beta = p
    # state variables
    h = u[1]
    q = u[2]

    # Jacobian
    J[1, 1] = lambda * (1 - 2 * h / kappa) - alpha * q
    J[1, 2] = -alpha * h
    J[1, 3] = 0.0
    J[2, 1] = -alpha * q
    J[2, 2] = beta * (1 - 2 * q / kappa) - alpha * h
    J[2, 3] = 0.0
    J[3, 1] = 1.0
    J[3, 2] = 0.0
    J[3, 3] = 0.0
    nothing
end

# Hazard-Response model with explicit Jacobian
HRJ = ODEFunction(HazResp; jac=jacHR)

#=
****************************************************************************
****************************************************************************
Hazard-Response model (log h and log q)
****************************************************************************
****************************************************************************
=#

# Hazard-Response ODE model
function HazRespL(dlh, lh, p, t)
    # Model parameters
    lambda, kappa, alpha, beta = p

    # ODE System
    dlh[1] = lambda * (1 - exp(lh[1]) / kappa) - alpha * exp(lh[2]) # log hazard
    dlh[2] = beta * (1 - exp(lh[2]) / kappa) - alpha * exp(lh[1]) # log response
    dlh[3] = exp(lh[1]) # cumulative hazard
    return nothing
end

# Jacobian for Hazard-Response model

function jacHRL(J, u, p, t)
    # Parameters
    lambda, kappa, alpha, beta = p
    # state variables
    lh = u[1]
    lq = u[2]

    # Jacobian
    J[1, 1] = -lambda * exp(lh) / kappa
    J[1, 2] = -alpha * exp(lq)
    J[1, 3] = 0.0
    J[2, 1] = -alpha * exp(lh)
    J[2, 2] = -beta * exp(lq) / kappa
    J[2, 3] = 0.0
    J[3, 1] = exp(lh)
    J[3, 2] = 0.0
    J[3, 3] = 0.0
    nothing
end

# Hazard-Response model with explicit Jacobian
HRJL = ODEFunction(HazRespL; jac=jacHRL)


#=
****************************************************************************
****************************************************************************
Log-likelihood and Negative log-likelihood functions: regression models
****************************************************************************
****************************************************************************
=#

# Negative log likelihood function (h and q)
mlog_lik = function (par::Vector{Float64})
    # if any(par .> 5.0)
    #     mloglik = Inf64
    # else
    # Parameters for the ODE
    odeparams = exp.(hcat(des_l * par[1:p_l],
        des_k * par[(p_l+1):(p_l+p_k)],
        des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
        des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


    OUT = zeros(n, 3)
    for i in 1:n
        sol = solve(ODEProblem(HRJ, u0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
        #    sol = solve(ODEProblem(HRJ, u0, tspan0[i, :], odeparams[i, :]), Tsit5())
        OUT[i, :] = reduce(vcat, sol.u[end, :])
    end

    # Terms in the log log likelihood function
    ll_haz = sum(log.(OUT[status, 1]))

    ll_chaz = sum(OUT[:, 3])

    mloglik = -ll_haz + ll_chaz

    return mloglik
end

# Negative log likelihood function (log h and log q)
mlog_likL = function (par)
    # if any(par .> 5.0)
    #     mloglik = Inf64
    # else
    # Parameters for the ODE
    odeparams = exp.(hcat(des_l * par[1:p_l],
        des_k * par[(p_l+1):(p_l+p_k)],
        des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
        des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


    OUT = zeros(Float64, n, 3)
    for i in 1:n
        sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
        #  sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]),Tsit5())
        OUT[i, :] = reduce(vcat, sol.u[end, :])
    end

    # Terms in the log log likelihood function
    ll_haz = sum(OUT[status, 1])

    ll_chaz = sum(OUT[:, 3])

    mloglik = -ll_haz + ll_chaz
    #   end
    return mloglik
end

# Negative log likelihood function (log h and log q): multi threading
mlog_likLMT = function (par)
    # if any(par .> 5.0)
    #     mloglik = Inf64
    # else
    # Parameters for the ODE
    odeparams = exp.(hcat(des_l * par[1:p_l],
        des_k * par[(p_l+1):(p_l+p_k)],
        des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
        des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


    OUT = zeros(Float64, n, 3)
    Threads.@threads :static for i in 1:n
        sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
        #    sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]),Tsit5())
        OUT[i, :] = reduce(vcat, sol.u[end, :])
    end

    # Terms in the log log likelihood function
    ll_haz = sum(OUT[status, 1])

    ll_chaz = sum(OUT[:, 3])

    mloglik = -ll_haz + ll_chaz
    #   end
    return mloglik
end


# log likelihood function (log h and log q)
log_likL = function (par)
    # if any(par .> 5.0)
    #     mloglik = Inf64
    # else
    # Parameters for the ODE
    odeparams = exp.(hcat(des_l * par[1:p_l],
        des_k * par[(p_l+1):(p_l+p_k)],
        des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
        des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


    OUT = zeros(Real, n, 3)
    for i in 1:n
        sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
        #  sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]),Tsit5())
        OUT[i, :] = reduce(vcat, sol.u[end, :])
    end

    # Terms in the log log likelihood function
    ll_haz = sum(OUT[status, 1])

    ll_chaz = sum(OUT[:, 3])

    loglik = ll_haz - ll_chaz
    #   end
    return loglik
end

# log likelihood function (log h and log q)
log_likLMT = function (par)
    # if any(par .> 5.0)
    #     mloglik = Inf64
    # else
    # Parameters for the ODE
    odeparams = exp.(hcat(des_l * par[1:p_l],
        des_k * par[(p_l+1):(p_l+p_k)],
        des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
        des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


    OUT = zeros(Real, n, 3)
    Threads.@threads for i in 1:n
        sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
        #  sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]),Tsit5())
        OUT[i, :] = reduce(vcat, sol.u[end, :])
    end

    # Terms in the log log likelihood function
    ll_haz = sum(OUT[status, 1])

    ll_chaz = sum(OUT[:, 3])

    loglik = ll_haz - ll_chaz
    #   end
    return loglik
end

#=
****************************************************************************
****************************************************************************
Log prior functions
****************************************************************************
****************************************************************************
=#

# log g-prior for slopes in variable selection
function log_gprior(beta, X, g)

    px = size(X, 2)

    mu = zeros(px)
    Sigma = g * Hermitian(inv(X' * X))
    distp = MvNormal(mu, Sigma)

    out = logpdf(distp, beta)
    return out
end

# Prior distribution for inference on the intercepts
distpriorint = Normal(0.0, 3.0)

# Prior distribution for inference on the slopes
distpriorbeta = Normal(0.0, 3.0)

# Prior distribution for the intercepts in variable selection
distpriorintvs = Normal(0.0, 3.0)

# Weak Prior distribution for inference 
distpriorbetaw = Normal(0.0, 10.0)

#=
****************************************************************************
Negative log-likelihood functions: no covariates
****************************************************************************
****************************************************************************
=#

# Negative log likelihood function
mlog_lik0 = function (par::Vector{Float64})
    if any(par .> 5.0)
        mloglik = Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(par)

        sol = solve(ODEProblem(HRJ, u0, [0.0, tmax], odeparams); alg_hints=[:stiff])
        #     sol = solve(ODEProblem(HRJ, u0, tspan0[i, :], odeparams[i, :]), Tsit5())
        OUT = sol(df.time)


        # Terms in the log log likelihood function
        ll_haz = sum(log.(OUT[1, status]))

        ll_chaz = sum(OUT[3, :])

        mloglik = -ll_haz + ll_chaz
    end
    return mloglik
end


# Negative log likelihood function (log h and log q)
mlog_likL0 = function (par::Vector{Float64})
    if any(par .> 5.0)
        mloglik = Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(par)


        sol = solve(ODEProblem(HRJL, lu0, [0.0, tmax], odeparams); alg_hints=[:stiff])
        #  sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]),Tsit5())
        OUT = sol(df.time)


        # Terms in the log log likelihood function
        ll_haz = sum(OUT[1, status])

        ll_chaz = sum(OUT[3, :])


        mloglik = -ll_haz + ll_chaz
    end
    return mloglik
end



#=
********************************************************************************************************************************************************
********************************************************************************************************************************************************
Function to find the MLE 
des_full: n x p matrix with all covariates to be considered in the model.
index_sel : p x p matrix with variable indicators for each linear predictor in each row.
M: number of iterations in the optimiser
init: initial value for optimiser
********************************************************************************************************************************************************
********************************************************************************************************************************************************
=#


function HRMLE(des_full, index_sel, M, init)
    # Index in Boolean format
    index_sel = collect(Bool, (index_sel))
    nvs = size(des_full, 1)

    # Design matrices
    des = hcat(ones(nvs), des_full)
    des_ls = des[:, index_sel[1, :]]
    des_ks = des[:, index_sel[2, :]]
    des_as = des[:, index_sel[3, :]]
    des_bs = des[:, index_sel[4, :]]

    # Number of parameters for each linear predictor
    p_ls = size(des_ls, 2)
    p_ks = size(des_ks, 2)
    p_as = size(des_as, 2)
    p_bs = size(des_bs, 2)

    # Log-posterior (log h and log q) for BVS: Multi-threading
    log_likLMT_index = function (par)

        # Parameters for the ODE
        odeparams = exp.(hcat(des_ls * par[1:p_ls],
            des_ks * par[(p_ls+1):(p_ls+p_ks)],
            des_as * par[(p_ls+p_ks+1):(p_ls+p_ks+p_as)],
            des_bs * par[(p_ls+p_ks+p_as+1):(p_ls+p_ks+p_as+p_bs)]))


        OUT = zeros(Real, n, 3)
        Threads.@threads for i in 1:n
            sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            # sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]), CVODE_BDF())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(OUT[status, 1])

        ll_chaz = sum(OUT[:, 3])


        loglik = ll_haz - ll_chaz

        return loglik
    end

    mlog_likLMT_index = function (par)

        out = -log_likLMT_index(par)
        return out
    end

    # Optimisation step
    optimiser = optimize(mlog_likLMT_index, init, method=NelderMead(), iterations=M)

    return optimiser, mlog_likLMT_index

end


#=
****************************************************************************
****************************************************************************
Negative log-posterior functions: no covariates
****************************************************************************
****************************************************************************
=#

# Negative log likelihood function
mlog_post0 = function (par::Vector{Float64})
    if any(par .> 5.0)
        mloglik = Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(par)

        sol = solve(ODEProblem(HRJ, u0, [0.0, tmax], odeparams); alg_hints=[:stiff])
        #     sol = solve(ODEProblem(HRJ, u0, tspan0[i, :], odeparams[i, :]), Tsit5())
        OUT = sol(df.time)


        # Terms in the log log likelihood function
        ll_haz = sum(log.(OUT[1, status]))

        ll_chaz = sum(OUT[3, :])

        mloglik = -ll_haz + ll_chaz

        mlogprior = -sum(logpdf.(distpriorintvs, par))

        mlogpost = mloglik + mlogprior
    end
    return mlogpost
end


# Negative log likelihood function (log h and log q)
mlog_postL0 = function (par::Vector{Float64})
    if any(par .> 5.0)
        mloglik = Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(par)


        sol = solve(ODEProblem(HRJL, lu0, [0.0, tmax], odeparams); alg_hints=[:stiff])
        #  sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]),Tsit5())
        OUT = sol(df.time)


        # Terms in the log log likelihood function
        ll_haz = sum(OUT[1, status])

        ll_chaz = sum(OUT[3, :])

        mloglik = -ll_haz + ll_chaz

        mlogprior = -sum(logpdf.(distpriorintvs, par))

        mlogpost = mloglik + mlogprior
    end
    return mlogpost
end

#=
****************************************************************************
****************************************************************************
Log-posterior and Negative log-posterior functions for inference
****************************************************************************
****************************************************************************
=#

# Negative Log-posterior (log h and log q): Multi-threading
mlog_postLMT = function (par)
    if any(par .> 5.0)
        lp = -Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(hcat(des_l * par[1:p_l],
            des_k * par[(p_l+1):(p_l+p_k)],
            des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
            des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


        OUT = zeros(Real, n, 3)
        Threads.@threads for i in 1:n
            sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            # sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]), CVODE_BDF())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(OUT[status, 1])

        ll_chaz = sum(OUT[:, 3])

        l_priorint = sum(logpdf.(distpriorint, par[indint]))

        l_priorbeta = sum(logpdf.(distpriorbeta, par[indbeta]))

        logpost = ll_haz - ll_chaz + l_priorint + l_priorbeta
    end
    return -logpost
end



# Negative Log-posterior (log h and log q): Multi-threading
mlog_postLMTW = function (par)
    if any(par .> 5.0)
        lp = -Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(hcat(des_l * par[1:p_l],
            des_k * par[(p_l+1):(p_l+p_k)],
            des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
            des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


        OUT = zeros(Real, n, 3)
        Threads.@threads for i in 1:n
            sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            # sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]), CVODE_BDF())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(OUT[status, 1])

        ll_chaz = sum(OUT[:, 3])

        l_prior = sum(logpdf.(distpriorbetaw, par))

        logpost = ll_haz - ll_chaz + l_prior
    end
    return -logpost
end


# Log-posterior (h and q)
log_post = function (par)
    if any(par .> 5.0)
        lp = -Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(hcat(des_l * par[1:p_l],
            des_k * par[(p_l+1):(p_l+p_k)],
            des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
            des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


        OUT = zeros(Float64, n, 3)
        for i in 1:n
            sol = solve(ODEProblem(HRJ, u0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            #   sol = solve(ODEProblem(HRJ, u0, tspan0[i, :], odeparams[i, :]),Tsit5())
            #   sol = solve(ODEProblem(HRJ, u0, tspan0[i, :], odeparams[i, :]),Rodas5P())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(log.(OUT[status, 1]))

        ll_chaz = sum(OUT[:, 3])

        l_priorint = sum(logpdf.(distpriorint, par[indint]))

        l_priorbeta = sum(logpdf.(distpriorbeta, par[indbeta]))

        logpost = ll_haz - ll_chaz + l_priorint + l_priorbeta
    end
    return logpost
end


# Log-posterior (log h and log q)
log_postL = function (par)
    if any(par .> 3.0)
        lp = -Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(hcat(des_l * par[1:p_l],
            des_k * par[(p_l+1):(p_l+p_k)],
            des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
            des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


        OUT = zeros(Float64, n, 3)
        for i in 1:n
            sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            # sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]), CVODE_BDF())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(OUT[status, 1])

        ll_chaz = sum(OUT[:, 3])

        l_priorint = sum(logpdf.(distpriorint, par[indint]))

        l_priorbeta = sum(logpdf.(distpriorbeta, par[indbeta]))

        logpost = ll_haz - ll_chaz + l_priorint + l_priorbeta
    end
    return logpost
end

# Log-posterior (log h and log q): Multi-threading
log_postLMT = function (par)
    if any(par .> 6.0)
        lp = -Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(hcat(des_l * par[1:p_l],
            des_k * par[(p_l+1):(p_l+p_k)],
            des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
            des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


        OUT = zeros(Float64, n, 3)
        Threads.@threads for i in 1:n
            sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            # sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]), CVODE_BDF())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(OUT[status, 1])

        ll_chaz = sum(OUT[:, 3])

        l_priorint = sum(logpdf.(distpriorint, par[indint]))

        l_priorbeta = sum(logpdf.(distpriorbeta, par[indbeta]))

        logpost = ll_haz - ll_chaz + l_priorint + l_priorbeta
    end
    return logpost
end


# Log-posterior (log h and log q): Multi-threading
log_postLMTW = function (par)
    if any(par .> 6.0)
        lp = -Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(hcat(des_l * par[1:p_l],
            des_k * par[(p_l+1):(p_l+p_k)],
            des_a * par[(p_l+p_k+1):(p_l+p_k+p_a)],
            des_b * par[(p_l+p_k+p_a+1):(p_l+p_k+p_a+p_b)]))


        OUT = zeros(Float64, n, 3)
        Threads.@threads for i in 1:n
            sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            # sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]), CVODE_BDF())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(OUT[status, 1])

        ll_chaz = sum(OUT[:, 3])

        l_prior = sum(logpdf.(distpriorbetaw, par))

        logpost = ll_haz - ll_chaz + l_prior
    end
    return logpost
end



#=
********************************************************************************************************************************************************
********************************************************************************************************************************************************
Function to find the MAP 
des_full: n x p matrix with all covariates to be considered in the model.
index_sel : p x p matrix with variable indicators for each linear predictor in each row.
gs : p x 1 vector with the values of the g hyperparameter to be used in the g-prior for each linear predictor.
M: number of iterations in the optimiser
init: initial value for optimiser
********************************************************************************************************************************************************
********************************************************************************************************************************************************
=#


function HRMAP(des_full, index_sel, gs, M, init)
    # Index in Boolean format
    index_sel = collect(Bool, (index_sel))
    nvs = size(des_full, 1)

    # Design matrices
    des = hcat(ones(nvs), des_full)
    des_ls = des[:, index_sel[1, :]]
    des_ks = des[:, index_sel[2, :]]
    des_as = des[:, index_sel[3, :]]
    des_bs = des[:, index_sel[4, :]]

    # Number of parameters for each linear predictor
    p_ls = size(des_ls, 2)
    p_ks = size(des_ks, 2)
    p_as = size(des_as, 2)
    p_bs = size(des_bs, 2)

    # Intercept positions
    indintvs = [1, p_ls + 1, p_ls + p_ks + 1, p_ls + p_ks + p_as + 1]

    # Log-posterior (log h and log q) for BVS: Multi-threading
    log_postLMT_index = function (par)

        # Parameters for the ODE
        odeparams = exp.(hcat(des_ls * par[1:p_ls],
            des_ks * par[(p_ls+1):(p_ls+p_ks)],
            des_as * par[(p_ls+p_ks+1):(p_ls+p_ks+p_as)],
            des_bs * par[(p_ls+p_ks+p_as+1):(p_ls+p_ks+p_as+p_bs)]))


        OUT = zeros(Real, n, 3)
        Threads.@threads for i in 1:n
            sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            # sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]), CVODE_BDF())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(OUT[status, 1])

        ll_chaz = sum(OUT[:, 3])

        # Log-prior for intercepts
        l_priorint = sum(logpdf.(distpriorintvs, par[indintvs]))

        # Log-g-prior for slopes
        if p_ls > 1
            lp_b_ls = log_gprior(par[2:p_ls], des_ls[:, 2:end], gs[1])
        else
            lp_b_ls = 0
        end

        if p_ks > 1
            lp_b_ks = log_gprior(par[(p_ls+2):(p_ls+p_ks)], des_ks[:, 2:end], gs[2])
        else
            lp_b_ks = 0
        end

        if p_as > 1
            lp_b_as = log_gprior(par[(p_ls+p_ks+2):(p_ls+p_ks+p_as)], des_as[:, 2:end], gs[3])
        else
            lp_b_as = 0
        end

        if p_bs > 1
            lp_b_bs = log_gprior(par[(p_ls+p_ks+p_as+2):(p_ls+p_ks+p_as+p_bs)], des_bs[:, 2:end], gs[4])
        else
            lp_b_bs = 0
        end

        l_priorbeta = lp_b_ls + lp_b_ks + lp_b_as + lp_b_bs

        logpost = ll_haz - ll_chaz + l_priorint + l_priorbeta

        return logpost
    end

    mlog_postLMT_index = function (par)

        out = -log_postLMT_index(par)
        return out
    end

    # Optimisation step
    optimiser = optimize(mlog_postLMT_index, init, method=NelderMead(), iterations=M)

    return optimiser, mlog_postLMT_index

end

#=
********************************************************************************************************************************************************
********************************************************************************************************************************************************
Laplace approximation function with multi-threading
des_full: n x p matrix with all covariates to be considered in the model.
index_sel : p x p matrix with variable indicators for each linear predictor in each row.
gs : p x 1 vector with the values of the g hyperparameter to be used in the g-prior for each linear predictor.
M: number of iterations in the optimiser
initvs: initial point for optimiser
********************************************************************************************************************************************************
********************************************************************************************************************************************************
=#


function Lap_ApproxMT(des_full, index_sel, gs, M)
    # Index in Boolean format
    index_sel = collect(Bool, (index_sel))
    nvs = size(des_full, 1)

    # Design matrices
    des = hcat(ones(nvs), des_full)
    des_ls = des[:, index_sel[1, :]]
    des_ks = des[:, index_sel[2, :]]
    des_as = des[:, index_sel[3, :]]
    des_bs = des[:, index_sel[4, :]]

    # Number of parameters for each linear predictor
    p_ls = size(des_ls, 2)
    p_ks = size(des_ks, 2)
    p_as = size(des_as, 2)
    p_bs = size(des_bs, 2)

    # Intercept positions
    indintvs = [1, p_ls + 1, p_ls + p_ks + 1, p_ls + p_ks + p_as + 1]

    # Log-posterior (log h and log q) for BVS: Multi-threading
    log_postLMT_BVS = function (par)

        # Parameters for the ODE
        odeparams = exp.(hcat(des_ls * par[1:p_ls],
            des_ks * par[(p_ls+1):(p_ls+p_ks)],
            des_as * par[(p_ls+p_ks+1):(p_ls+p_ks+p_as)],
            des_bs * par[(p_ls+p_ks+p_as+1):(p_ls+p_ks+p_as+p_bs)]))


        OUT = zeros(Real, n, 3)
        Threads.@threads for i in 1:n
            sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            # sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]), CVODE_BDF())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(OUT[status, 1])

        ll_chaz = sum(OUT[:, 3])

        # Log-prior for intercepts
        l_priorint = sum(logpdf.(distpriorintvs, par[indintvs]))

        # Log-g-prior for slopes
        if p_ls > 1
            lp_b_ls = log_gprior(par[2:p_ls], des_ls[:, 2:end], gs[1])
        else
            lp_b_ls = 0
        end

        if p_ks > 1
            lp_b_ks = log_gprior(par[(p_ls+2):(p_ls+p_ks)], des_ks[:, 2:end], gs[2])
        else
            lp_b_ks = 0
        end

        if p_as > 1
            lp_b_as = log_gprior(par[(p_ls+p_ks+2):(p_ls+p_ks+p_as)], des_as[:, 2:end], gs[3])
        else
            lp_b_as = 0
        end

        if p_bs > 1
            lp_b_bs = log_gprior(par[(p_ls+p_ks+p_as+2):(p_ls+p_ks+p_as+p_bs)], des_bs[:, 2:end], gs[4])
        else
            lp_b_bs = 0
        end

        l_priorbeta = lp_b_ls + lp_b_ks + lp_b_as + lp_b_bs

        logpost = ll_haz - ll_chaz + l_priorint + l_priorbeta

        return logpost
    end

    mlog_postLMT_BVS = function (par)

        out = -log_postLMT_BVS(par)
        return out
    end

    # Initial values for optimisation
    #initvs = zeros(p_ls + p_ks + p_as + p_bs)
    #initvs[indintvs] = MAP0
    initvs = OPT[1].minimizer[findall(reshape(index_sel', :) .> 0)]

    # Optimisation step
    optimiserBVS = optimize(mlog_postLMT_BVS, initvs, method=NelderMead(), iterations=M)

    # MAP and Hessian
    MAPR_BVS = optimiserBVS.minimizer

    HESS_BVS = Hermitian(ForwardDiff.hessian(mlog_postLMT_BVS, MAPR_BVS))

    if (det(HESS_BVS) < 0.0)
        # Optimisation step
        optimiserBVS = optimize(mlog_postLMT_BVS, initvs .* 0.0, method=NelderMead(), iterations=M)
        # MAP and Hessian
        MAPR_BVS = optimiserBVS.minimizer

        HESS_BVS = Hermitian(ForwardDiff.hessian(mlog_postLMT_BVS, MAPR_BVS))
    end

    if (det(HESS_BVS) > 0.0)

        # Laplace approximation of the log marginal likelihood
        logML = -optimiserBVS.minimum + 0.5 * length(MAPR_BVS) * log(2 * pi) - 0.5 * logdet(HESS_BVS)

    else
        logML = -Inf
    end

    return logML

end

#=
********************************************************************************************************************************************************
********************************************************************************************************************************************************
# Complexity prior on gamma
# Unnormalised
# index_sel: variable inclusion indicators
# lambda: hyper-parameter
# p: total number of variables
********************************************************************************************************************************************************
********************************************************************************************************************************************************
=#

function prior_gamma(index_sel, lambda, p)
    out = exp(-lambda*sum(index_sel)*log(p))
return out
end

#=
********************************************************************************************************************************************************
********************************************************************************************************************************************************
# Gibbs sampler function using Laplace approximation
seed: seed for RNG 
niter: number of iterations in the Gibbs sampler
init_ind: initial model. p x p matrix with variable indicators for each linear predictor in each row.
des_full: n x p matrix with all covariates to be considered in the model.
gs : p x 1 vector with the values of the g hyperparameter to be used in the g-prior for each linear predictor.
burnin: burn-in period
M: number of iterations in the optimiser
********************************************************************************************************************************************************
********************************************************************************************************************************************************
=#

#=
Structure for output of the Gibbs sampler based on LA
=#
struct Gibbs_output
    chain::Array{Int64, 3}
    modpp::OrderedDict{String,Float64}
    #    modpp_ml::Vector{Float64}
    #    vismod::Array{Int64, 3}
    pip::Array{Float64,3}
end



# Gibbs sampler function
function Gibbs_varsel(seed, niter, init_ind, des_full, gs, burnin, M)
    print("Initialising ...")
    Random.seed!(seed)
    p = size(init_ind, 1)
    q = size(init_ind, 2)
    chain = zeros(Int, niter, p, q)
    curr_ind = copy(init_ind)
    burnin = burnin + 1

    # Ensure the intercept is always included
    curr_ind[:, 1] .= 1  # Set the first column to 1

    # Cache for storing log-marginal likelihoods
    cache = Dict{String,Float64}()

    # Dictionary to store unique models and their log-marginal likelihoods
    unique_models_dict = Dict{String,Float64}()

    # Helper function to convert inclusion matrix to a string key
    function ind_to_key(ind)
        return join(string.(ind[:]), "_")  # Flatten and convert to string
    end

    # Compute Laplace approximation for the initial model and cache it
    curr_key = ind_to_key(curr_ind)
    cache[curr_key] = Lap_ApproxMT(des_full, curr_ind, gs, M)
    curr_log_ml = cache[curr_key]

    # Store the initial model in the unique models dictionary
    unique_models_dict[curr_key] = curr_log_ml

    for t in 1:niter
        percent = round(t / niter * 100; digits=1)
        print("\rProgress: $percent%")  # Overwrites the same line
        flush(stdout)  # Ensures real-time printing

        for j in 2:q  # Skip the first column (j = 1)
            for i in 1:p
                #     print([i, j])
                # Propose new inclusion matrix
                prop_ind = copy(curr_ind)
                prop_ind[i, j] = 1 - prop_ind[i, j]
                println(prop_ind)
                #               println(prop_ind)
                # Check if the proposed model is already in the cache
                prop_key = ind_to_key(prop_ind)
                if haskey(cache, prop_key)
                    prop_log_ml = cache[prop_key]
                else
                    # Compute Laplace approximation and cache it
                    prop_log_ml = Lap_ApproxMT(des_full, prop_ind, gs, M)
                    cache[prop_key] = prop_log_ml
                end

                # Store the proposed model in the unique models dictionary
                if !haskey(unique_models_dict, prop_key)
                    unique_models_dict[prop_key] = prop_log_ml
                end

                # Acceptance probability
                w_prop = prior_gamma(prop_ind, 0.1, 16.0)
                w_curr = prior_gamma(curr_ind, 0.1, 16.0)
                α = min(1.0, wexp_ratio(prop_log_ml, curr_log_ml, w_prop, w_curr))

                # Accept or reject
                if rand() < α
                    curr_ind = prop_ind
                    curr_log_ml = prop_log_ml
                end
                #             println([i, j])

            end
        end
        chain[t, :, :] = curr_ind
    end

    # Post-processing
    println("Post-processing ...")
    # Total number of iterations after burn-in
    npost = niter - burnin
    # Convert each model in the chain to a string key
    chain_keys = [ind_to_key(chain[t, :, :]) for t in burnin:niter]

    # Count the frequency of each unique model
    model_counts = Dict{String,Int}()
    for key in chain_keys
        model_counts[key] = get(model_counts, key, 0) + 1
    end

    # Compute posterior model probabilities
    #    post_model_probs = Dict(key => count / niter for (key, count) in model_counts)
    post_model_probs = Dict(key => count / npost for (key, count) in model_counts)

    sorted_pairs = sort(collect(post_model_probs), by=x -> x[2], rev=true)

    # Step 2: If you want to convert it back to a Dict (note: Dict is unordered)
    sorted_post_model_probs = OrderedDict(sorted_pairs)

    # Compute posterior inclusion probabilities
    post_inclusion_probs = mean(chain[burnin:niter, :, :], dims=1)  # Posterior inclusion probabilities

    return Gibbs_output(chain,sorted_post_model_probs, post_inclusion_probs)
end





#=
********************************************************************************************************************************************************
********************************************************************************************************************************************************
Function to find the MAP (Weakly informative priors)
des_full: n x p matrix with all covariates to be considered in the model.
index_sel : p x p matrix with variable indicators for each linear predictor in each row.
gs : p x 1 vector with the values of the g hyperparameter to be used in the g-prior for each linear predictor.
M: number of iterations in the optimiser
init: initial value for optimiser
********************************************************************************************************************************************************
********************************************************************************************************************************************************
=#



function HRMAPW(des_full, times, status, index_sel, M, init)
    # Index in Boolean format
    index_sel = collect(Bool, (index_sel))
    nvs = size(des_full, 1)

        # Time grids
        tspan0 = hcat(zeros(n), times)

        status = collect(Bool, status)

    # Design matrices
    des = hcat(ones(nvs), des_full)
    des_ls = des[:, index_sel[1, :]]
    des_ks = des[:, index_sel[2, :]]
    des_as = des[:, index_sel[3, :]]
    des_bs = des[:, index_sel[4, :]]

    # Number of parameters for each linear predictor
    p_ls = size(des_ls, 2)
    p_ks = size(des_ks, 2)
    p_as = size(des_as, 2)
    p_bs = size(des_bs, 2)


    # Log-posterior (log h and log q) for BVS: Multi-threading
    log_postLMT_index = function (par)

        # Parameters for the ODE
        odeparams = exp.(hcat(des_ls * par[1:p_ls],
            des_ks * par[(p_ls+1):(p_ls+p_ks)],
            des_as * par[(p_ls+p_ks+1):(p_ls+p_ks+p_as)],
            des_bs * par[(p_ls+p_ks+p_as+1):(p_ls+p_ks+p_as+p_bs)]))


        OUT = zeros(Real, n, 3)
        #for i in 1:n
            Threads.@threads for i in 1:n
            sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]); alg_hints=[:stiff])
            # sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]), CVODE_BDF())
            OUT[i, :] = reduce(vcat, sol.u[end, :])
        end

        # Terms in the log log likelihood function
        ll_haz = sum(OUT[status, 1])

        ll_chaz = sum(OUT[:, 3])

        # Log-prior for regression coefficients
        l_priorbeta = sum(logpdf.(distpriorbetaw, par))

        logpost = ll_haz - ll_chaz + l_priorbeta

        return logpost
    end

    mlog_postLMT_index = function (par)

        out = -log_postLMT_index(par)
        return out
    end

    # Optimisation step
    optimiser = optimize(mlog_postLMT_index, init, method=NelderMead(), iterations=M)

    return optimiser, mlog_postLMT_index

end