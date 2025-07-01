# Quantile functions
ql <- function(p) quantile(p, 0.025)
qu <- function(p) quantile(p, 0.975)


######################################################################################
######################################################################################
######################################################################################
# Weibull
######################################################################################
######################################################################################
######################################################################################


#--------------------------------------------------------------------------------------------------
# Weibull -log-posterior function
#--------------------------------------------------------------------------------------------------

log_postW <- function(par){
  sigma <- exp(par[1]); nu <- exp(par[2]);
  
  # Terms in the log log likelihood function
  ll_haz <- sum(hweibull(t_obs, sigma, nu, log = TRUE))
  
  ll_chaz <- sum(chweibull(survtimes, sigma, nu))
  
  log_lik <- -ll_haz + ll_chaz 
  
  # Log prior
  
  log_prior <- -dgamma(sigma, shape = 2, scale = 2, log = TRUE) - 
    dgamma(nu, shape = 2, scale = 2, log = TRUE) 
  
  # Log Jacobian
  
  log_jacobian <- -par[1] - par[2] 
  
  # log posterior
  
  log_post0 <- log_lik + log_prior + log_jacobian
  
  return(as.numeric(log_post0))
}



#--------------------------------------------------------------------------------------------------
# 2-Arm Weibull MLE
#--------------------------------------------------------------------------------------------------

W2MLE <- function(init, times, status, arm, method, control){
  
  times = as.vector(times)
  status = as.logical(status)
  arm = as.logical(arm)

  timesobs_arm0 <- times[status == 1 & arm == 0]  # Times for status 1 and arm 0
  timesobs_arm1 <- times[status == 1 & arm == 1]  # Times for status 1 and arm 1
  
  times_arm0 <- times[arm == 0]  # Times for arm 0
  times_arm1 <- times[arm == 1]  # Times for arm 1
  
  n <- length(times)
  
  
  loglikW2 <- function(par){
    sigma0 <- exp(par[1]); nu0 <- exp(par[2]);
    sigma1 <- exp(par[3]); nu1 <- exp(par[4]);
    
    # Terms in the log log likelihood function
    ll_haz0 <- sum(hweibull(timesobs_arm0, sigma0, nu0, log = TRUE))
    
    ll_chaz0 <- sum(chweibull(times_arm0, sigma0, nu0))
    
    ll_haz1 <- sum(hweibull(timesobs_arm1, sigma1, nu1, log = TRUE))
    
    ll_chaz1 <- sum(chweibull(times_arm1, sigma1, nu1))
    
    log_lik <- -ll_haz0 + ll_chaz0  -ll_haz1 + ll_chaz1
    
    
    return(as.numeric(log_lik))
  }
  
  
  # Optimisation step
  if (method != "nlminb") {
    OPT <- optim(init, loglikW2, control = control, 
                 method = method, hessian = TRUE)
    DEV <- OPT$value 
  }
  if (method == "nlminb") {
    OPT <- nlminb(init, loglikW2, control = control)
    DEV <- OPT$objective 
  }
  
  AIC <- 2*DEV + 2*4
  BIC <- 2*DEV + log(n)*4
  
  MLE <- as.vector(exp(OPT$par))
  names(MLE) <- c("sigma_0","nu_0","sigma_1","nu_1")
  
  # Output
  OUT <- list(loglik = loglikW2, OPT = OPT, MLE = MLE, AIC = AIC, BIC = BIC)
  return(OUT)
  
}


######################################################################################
######################################################################################
######################################################################################
# Power Generalised Weibull
######################################################################################
######################################################################################
######################################################################################


#--------------------------------------------------------------------------------------------------
# Power Generalised Weibull -log-posterior function
#--------------------------------------------------------------------------------------------------

log_postPGW <- function(par){
  sigma <- exp(par[1]); nu <- exp(par[2]); gamma <- exp(par[3])
  
  # Terms in the log log likelihood function
  ll_haz <- sum(hpgw(t_obs, sigma, nu, gamma, log = TRUE))
  
  ll_chaz <- sum(chpgw(survtimes, sigma, nu, gamma))
  
  log_lik <- -ll_haz + ll_chaz 
  
  # Log prior
  
  log_prior <- -dgamma(sigma, shape = 2, scale = 2, log = TRUE) - 
    dgamma(nu, shape = 2, scale = 2, log = TRUE) - 
    dgamma(gamma, shape = 2, scale = 2, log = TRUE) 
  
  # Log Jacobian
  
  log_jacobian <- -par[1] - par[2] - par[3]
  
  # log posterior
  
  log_post0 <- log_lik + log_prior + log_jacobian
  
  return(as.numeric(log_post0))
}

######################################################################################
######################################################################################
######################################################################################
# Logistic ODE
######################################################################################
######################################################################################
######################################################################################

#-----------------------------------------------------------------------------
# Logistic ODE Hazard Function: Analytic solution
#-----------------------------------------------------------------------------
# t: time (positive)
# lambda: intrinsic growth rate (positive)
# kappa: upper bound (positive)
# h0: hazard initial value (positive)
hlogisode <- function(t, lambda, kappa, h0, log = FALSE){
  lhaz <-  log(kappa) + log(h0) - log( (kappa-h0)*exp(-lambda*t) + h0)
  if (log)  return(lhaz)
  else return(exp(lhaz))
}

#-----------------------------------------------------------------------------
# Logistic ODE Cumulative Hazard Function: Analytic solution
#-----------------------------------------------------------------------------
# t: time (positive)
# lambda: intrinsic growth rate (positive)
# kappa: upper bound (positive)
# h0: hazard initial value (positive)
chlogisode <- function(t, lambda, kappa, h0){
  chaz <-   kappa*( log( (kappa-h0)*exp(-lambda*t) + h0 ) - log(kappa) + lambda*t )/lambda
  return(chaz)
}

#-----------------------------------------------------------------------------
# Logistic ODE random number generation
#-----------------------------------------------------------------------------
# t: time (positive)
# lambda: intrinsic growth rate (positive)
# kappa: upper bound (positive)
# h0: hazard initial value (positive)
rlogisode <- function(n, lambda, kappa, h0){
  u <- runif(n)
  times <- log( 1 + kappa*( exp(-lambda*log(1-u)/kappa) -1 )/h0    )/lambda 
  return(as.vector(times))
}

#-----------------------------------------------------------------------------
# Logistic ODE Probability Density Function: Analytic solution
#-----------------------------------------------------------------------------
# t: time (positive)
# lambda: intrinsic growth rate (positive)
# kappa: upper bound (positive)
# h0: hazard initial value (positive)
dlogisode <- function(t, lambda, kappa, h0, log = FALSE){
  lden <-  hlogisode(t, lambda, kappa, h0, log = TRUE) - 
    chlogisode(t, lambda, kappa, h0)
  if (log)  return(lden)
  else return(exp(lden))
}



#--------------------------------------------------------------------------------------------------
# Logistic ODE -log-likelihood function: Analytic solution
#--------------------------------------------------------------------------------------------------

log_likL <- function(par){
  lambda <- exp(par[1]); kappa <- exp(par[2]); h0 <- exp(par[3])
    # Terms in the log log likelihood function
    ll_haz <- sum(hlogisode(t_obs, lambda, kappa, h0, log = TRUE))
    
    ll_chaz <- sum(chlogisode(survtimes, lambda, kappa, h0 ))
    
    log_lik <- -ll_haz + ll_chaz 
    
    return(log_lik)
}




#--------------------------------------------------------------------------------------------------
# Logistic ODE -log-posterior function: Analytic solution
#--------------------------------------------------------------------------------------------------

log_postL <- function(par){
  lambda <- exp(par[1]); kappa <- exp(par[2]); h0 <- exp(par[3])
  
  # Terms in the log log likelihood function
  ll_haz <- sum(hlogisode(t_obs, lambda, kappa, h0, log = TRUE))
  
  ll_chaz <- sum(chlogisode(survtimes, lambda, kappa, h0 ))
  
  log_lik <- -ll_haz + ll_chaz 
  
  # Log prior
  
  log_prior <- -dgamma(exp(par[1]), shape = 2, scale = 2, log = TRUE) - 
    dgamma(exp(par[2]), shape = 2, scale = 2, log = TRUE) -
    dgamma(exp(par[3]), shape = 2, scale = 2, log = TRUE) 
  
  # Log-Jacobian
  
  log_jacobian <- - par[1] - par[3] - par[3]
    
  # log posterior
  
  log_post0 <- log_lik + log_prior + log_jacobian

  return(as.numeric(log_post0))
}




######################################################################################
# Logistic ODE linear regression 
######################################################################################

#--------------------------------------------------------------------------------------------------
# Logistic ODE Regression -log-likelihood function: Analytic solution
#--------------------------------------------------------------------------------------------------

log_likreg2 <- function(par){
  lambda <- as.vector(exp(des_l%*%par[1:p_l])); 
  kappa <- as.vector(exp(des_k%*%par[(p_l+1):(p_l+p_k)])); 
  h0 <- exp(par[p_l+p_k+1])
  # Terms in the log log likelihood function
  ll_haz <- sum(hlogisode(t_obsr, lambda[statusr], kappa[statusr], rep(h0,nobs), log = TRUE))
  
  ll_chaz <- sum(chlogisode(survtimesr, lambda, kappa, rep(h0,n) ))
  
  log_lik <- -ll_haz + ll_chaz 
  
  return(log_lik)
}


#--------------------------------------------------------------------------------------------------
# Logistic ODE Regression -log-posterior function: Analytic solution
#--------------------------------------------------------------------------------------------------

log_postreg2 <- function(par){
  lambda <- as.vector(exp(des_l%*%par[1:p_l])); 
  kappa <- exp(des_k%*%par[(p_l+1):(p_l+p_k)]); 
  h0 <- exp(par[p_l+p_k+1])
  # Terms in the log log likelihood function
  ll_haz <- sum(hlogisode(t_obsr, lambda[statusr], kappa[statusr], rep(h0,nobs), log = TRUE))
  
  ll_chaz <- sum(chlogisode(survtimesr, lambda, kappa, rep(h0,n) ))
  
  log_lik <- -ll_haz + ll_chaz 
  
  log_prior <- - dnorm(par[1],0,10, log = TRUE) -
    sum(dnorm(par[2:p_l],0,10,log = TRUE)) -
    sum(dnorm(par[(p_l+1):(p_l+p_k)],0,10,log = TRUE)) -
    dgamma(h0, shape = 2, rate = 1/2, log = TRUE)
  
  log_post0 <- log_lik + log_prior
  
  return(log_post0)
}


LODERMLE <- function(init, times, status, arm, method, control){
  
 times = as.vector(times)
 status = as.logical(status)
 arm = as.logical(arm)
  
timesobs <- times[status]  # Times for status 1 

  n <- length(times)
  nobs <- length(timesobs)
  # design matrix
  des <- as.matrix(cbind(1,arm))
  p=2
  
  log_likreg <- function(par){
    lambda <- as.vector(exp(des%*%par[1:p])); 
    kappa <- as.vector(exp(des%*%par[(p+1):(2*p)])); 
    h0 <- exp(par[2*p+1])
    # Terms in the log log likelihood function
    ll_haz <- sum(hlogisode(timesobs, lambda[status], kappa[status], rep(h0,nobs), log = TRUE))
    
    ll_chaz <- sum(chlogisode(times, lambda, kappa, rep(h0,n) ))
    
    log_lik <- -ll_haz + ll_chaz 
    
    return(log_lik)
  }
  
  
  # Optimisation step
  if (method != "nlminb") {
    OPT <- optim(init, log_likreg, control = control, 
                 method = method, hessian = TRUE)
    DEV <- OPT$value 
  }
  if (method == "nlminb") {
    OPT <- nlminb(init, log_likreg, control = control)
    DEV <- OPT$objective 
  }
  
  AIC <- 2*DEV + 2*length(OPT$par)
  BIC <- 2*DEV + log(n)*length(OPT$par)
  
  MLE <- as.vector(c(OPT$par[1:4],exp(OPT$par[5])))
  names(MLE) <- c("alpha_0 (l)", "alpha_1 (l)", "beta_0 (k)", "beta_1 (k)", "h0")
  
  # Output
  OUT <- list(loglik = log_likreg, OPT = OPT, MLE = MLE, AIC = AIC, BIC = BIC)
  return(OUT)
  
}


