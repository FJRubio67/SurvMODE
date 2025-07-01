## ----message=FALSE--------------------------------------------------------------------------------------------------------------------------------

rm(list=ls())

# Required packages
library(deSolve)
library(survival)
library(ggplot2)
library(survminer)
#library(devtools)
#install_github("FJRubio67/HazReg")
library(HazReg)
library(Rtwalk)
library(knitr)
library(splines)
library(mvtnorm)
library("survminer")
library(spBayesSurv)
source("routinesL.R")


## -------------------------------------------------------------------------------------------------------------------------------------------------
# Reading data (CA184-043, NCT00861614, ipilimumab immunotherapy trial)
df0 <- read.csv("CA184043_2A.csv", header = TRUE)
head(df0)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# Data preparation
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


# New data frame: survival time, vital status, trt (ipilimumab)
trt0 <- vector()
trt0 <- ifelse(as.factor(df0$Arm) == "ipilimumab", 1 , 0)

df <- data.frame(time = df0$Time, status = df0$Event, trt = trt0)
df <- df[order(df$time),]

dim(df)
head(df)

# Required quantities
status <- as.logical(df$status)
survLTimes <- df$time


## -------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================== 
#================================================================================================== 
# Maximum Likelihood Regression Analysis 
#================================================================================================== 
#================================================================================================== 

#-------------------------------------------------------------------------------------------------- 
# Logistic ODE regression model for the hazard function
#-------------------------------------------------------------------------------------------------- 

# Initial point 
initr <- c(0,0,0,0,log(0.01)) 

# Optimisation step 
OPTR <-LODERMLE(init = initr, times = survLTimes, status = status, arm = df$trt, method = "nlminb",
      control = list(iter.max = 1000))

MLER <- OPTR$MLE

# Group 1: : intercept + no treatment
valx_lNT <- c(1,0) 
valx_kNT <- c(1,0) 

lambdaNT <- exp(valx_lNT%*%MLER[1:2])
kappaNT <- exp(valx_kNT%*%MLER[3:4])
h0 <- MLER[5]

# Fitted hazard and survival: Logistic model
hazLNT <- Vectorize(function(t) hlogisode(t, lambdaNT, kappaNT, h0))
survLNT <- Vectorize(function(t) exp(-chlogisode(t, lambdaNT, kappaNT, h0)))


# Group 2: intercept + treatment
valx_lT <- c(1,1) 
valx_kT <- c(1,1) 

lambdaT <- exp(valx_lT%*%MLER[1:2])
kappaT <- exp(valx_kT%*%MLER[3:4])

# Fitted hazard and survival: Logistic model
hazLT <- Vectorize(function(t) hlogisode(t, lambdaT, kappaT, h0))
survLT <- Vectorize(function(t) exp(-chlogisode(t, lambdaT, kappaT, h0)))



# Kaplan-Meier estimator 
km <- survfit(Surv(time, status) ~ trt, data = df)

plot(km, col = c("gray", "gray"), lty = c(2,1), lwd = 2,
     xlab = "Time (months)", ylab = "Fitted Survival", 
     xlim = c(0, 37), ylim = c(0,1), cex.lab = 1.5, cex.axis = 1.5)
# Fitted survival
curve(survLT, 0, max(survLTimes), ylim = c(0.0,1), lwd = 2, xlab = "Time (months)", ylab = "Survival", 
      cex.axis = 1.5, cex.lab = 1.5, add = TRUE)
curve(survLNT, 0, max(survLTimes), add=T, lty = 2, lwd = 2)
legend("topright", legend = c("Treatment","No Treatment"), lty = c(1,2), lwd = c(2,2),  
       col = c("black","black")) 


# Fitted hazard
curve(hazLT, 0, max(survLTimes), ylim = c(0, 0.1), lwd = 2, xlab = "Time (months)", ylab = "Hazard", 
      cex.axis = 1.5, cex.lab = 1.5)
curve(hazLNT, 0, max(survLTimes), add=T, lty = 2, lwd = 2)
legend("bottomright", legend = c("Treatment","No Treatment"), lty = c(1,2), lwd = c(2,2),  
       col = c("black","black")) 



## -------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================== 
#================================================================================================== 
# Maximum Likelihood Regression Analysis 
#================================================================================================== 
#================================================================================================== 

#-------------------------------------------------------------------------------------------------- 
# 2-Arm Weibull model
#-------------------------------------------------------------------------------------------------- 

# Initial point 
initW <- c(0,0,0,0) 

# Optimisation step 
OPTW <- W2MLE(init = initW, times = survLTimes, status = status, arm = df$trt, method = "nlminb",
              control = list(iter.max = 1000))

MLEW <- OPTW$MLE

# Comparison (Best model: logistic ODE)

c(OPTR$AIC, OPTW$AIC)
c(OPTR$BIC, OPTW$BIC)

# Group 1: : intercept + no treatment
hazWNT <- Vectorize(function(t) hweibull(t, MLEW[1], MLEW[2]))
survWNT <- Vectorize(function(t) exp(-chweibull(t, MLEW[1], MLEW[2])))


# Group 2: intercept + treatment
hazWT <- Vectorize(function(t) hweibull(t, MLEW[3], MLEW[4]))
survWT <- Vectorize(function(t) exp(-chweibull(t, MLEW[3], MLEW[4])))

# Fitted survival
curve(survWT, 0, max(survLTimes), ylim = c(0.0,1), lwd = 2, xlab = "Time (months)", ylab = "Survival", 
      cex.axis = 1.5, cex.lab = 1.5)
curve(survWNT, 0, max(survLTimes), add=T, lty = 2, lwd = 2)
legend("topright", legend = c("Treatment","No Treatment"), lty = c(1,2), lwd = c(2,2),  
       col = c("black","black")) 

# Fitted hazard
curve(hazWT, 0, max(survLTimes), ylim = c(0, 0.15), lwd = 2, xlab = "Time (months)", ylab = "Hazard", 
      cex.axis = 1.5, cex.lab = 1.5)
curve(hazWNT, 0, max(survLTimes), add=T, lty = 2, lwd = 2)
legend("bottomright", legend = c("Treatment","No Treatment"), lty = c(1,2), lwd = c(2,2),  
       col = c("black","black")) 



## -------------------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================== 
#================================================================================================== 
# Bayesian Analysis 
#================================================================================================== 
#================================================================================================== 

# Design matrices for the regression models for each parameter 
# lambda: intercept + treatment
des_l <- as.matrix(cbind(1,df$trt))
# kappa: intercept + treatment
des_k <- as.matrix(cbind(1,df$trt))


# Required quantities 
p_l <- ncol(des_l) 
p_k <- ncol(des_k)
statusr <- as.logical(df$status)
t_obsr <- df$time[status]
survtimesr <- df$time 
nobs <- sum(status)
n <- nrow(df)


#-------------------------------------------------------------------------------------------------- 
# Logistic ODE regression model for the hazard function: Analytic solution 
#-------------------------------------------------------------------------------------------------- 

# Support 
Support <- function(x) {  TRUE } 

# Random initial points 
X0R <- function(x) { OPTR$OPT$par + runif(length(OPTR$OPT$par),-0.1,0.1) } 

# twalk for analytic solution 
set.seed(123) 
infor <- Runtwalk( dim=length(OPTR$OPT$par),  Tr=55000,  Obj=log_postreg2, Supp=Support, x0=X0R(), xp0=X0R(), 
                   PlotLogPost = FALSE)  


# Posterior sample after burn-in and thinning 
ind=seq(5000,55000,50) 

# Summaries 
summr <- apply(infor$output[ind,],2,summary) 
colnames(summr) <- c("alpha0","alpha1","beta0","beta1","h_0") 
kable(summr, digits = 3) 

# KDEs 
alphap0 <- infor$output[,1][ind] 
alphap1 <- infor$output[,2][ind] 
betap0 <- infor$output[,3][ind] 
betap1 <- infor$output[,4][ind] 
h0p <- exp(infor$output[,5][ind]) 

plot(density(alphap0), main = "Posterior sample", xlab = "alpha_0", ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 
plot(density(alphap1), main = "Posterior sample", xlab = "alpha_1", ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 
plot(density(betap0), main = "Posterior sample", xlab = "beta_0", ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 
plot(density(betap1), main = "Posterior sample", xlab = "beta_1", ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 
plot(density(h0p), main = "Posterior sample", xlab = "h_0", ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 


#-------------------------------------------------------------------------------------------------- 
# Predictive logistic survival: Analytic solution 
#-------------------------------------------------------------------------------------------------- 

acoefs <- cbind(alphap0,alphap1) 
bcoefs <- cbind(betap0,betap1) 

# Predictive logistic survival: Treatment

predsLT <- Vectorize(function(t){ 
  out <- vector() 
  for(i in 1:length(ind)){ 
    lambda <- as.numeric(exp(valx_lT%*%acoefs[i,]))
    kappa <- as.numeric(exp(valx_kT%*%bcoefs[i,]))
    out[i] <- exp(-chlogisode( t,lambda, kappa, h0p[i]) )  
  } 
  return(mean(out)) 
}) 


# Predictive logistic survival: No Treatment

predsLNT <- Vectorize(function(t){ 
  out <- vector() 
  for(i in 1:length(ind)){ 
    lambda <- as.numeric(exp(valx_lNT%*%acoefs[i,]))
    kappa <- as.numeric(exp(valx_kNT%*%bcoefs[i,]))
    out[i] <- exp(-chlogisode( t,lambda, kappa, h0p[i]) )  
  } 
  return(mean(out)) 
}) 


# Comparison
plot(km, col = c("gray", "gray"), lty = c(2,1), lwd = 2,
     xlab = "Time (months)", ylab = "Predictive Survival", 
     xlim = c(0, 37), ylim = c(0,1), cex.lab = 1.5, cex.axis = 1.5)

curve(predsLT, 0, 37, ylim = c(0,1), xlab = "Time (months)", ylab = "Predictive Survival", lwd = 2, lty = 1, 
      cex.lab = 1.5, cex.axis = 1.5, add = TRUE) 
curve(predsLNT, 0, 37, ylim = c(0,1), lwd = 2, lty = 2, add = TRUE) 
legend("topright", legend = c("Treatment","No Treatment"), lty = c(1,2), lwd = c(2,2),  
       col = c("black","black")) 


#---------------------------------------------------------------------------------------
# Predictive Hazard functions
#---------------------------------------------------------------------------------------

# Predictive Logistic hazard: Treatment
predhLT <- Vectorize(function(t){
  num <- den <- vector()
  for(i in 1:length(ind)){
    lambda <- as.numeric(exp(valx_lT%*%acoefs[i,]))
    kappa <- as.numeric(exp(valx_kT%*%bcoefs[i,]))
    num[i] <- exp(-chlogisode( t, lambda, kappa, h0p[i]))*hlogisode( t, lambda, kappa, h0p[i])
    den[i] <- exp(-chlogisode( t, lambda, kappa, h0p[i]))
  } 
  return(mean(num)/mean(den))
})

# Predictive Logistic hazard: No Treatment
predhLNT <- Vectorize(function(t){
  num <- den <- vector()
  for(i in 1:length(ind)){
    lambda <- as.numeric(exp(valx_lNT%*%acoefs[i,]))
    kappa <- as.numeric(exp(valx_kNT%*%bcoefs[i,]))
    num[i] <- exp(-chlogisode( t, lambda, kappa, h0p[i]))*hlogisode( t, lambda, kappa, h0p[i])
    den[i] <- exp(-chlogisode( t, lambda, kappa, h0p[i]))
  } 
  return(mean(num)/mean(den))
})



# Logistic
curve(predhLT, 0, 37, ylim = c(0, 0.1), lwd = 2, xlab = "Time (months)", ylab = "Hazard", 
      cex.axis = 1.5, cex.lab = 1.5)
curve(predhLNT, 0, 37, add=T, lty = 2, lwd = 2)
legend("bottomright", legend = c("Treatment","No Treatment"), lty = c(1,2), lwd = c(2,2),  
       col = c("black","black")) 

# Zoom in
curve(predhLT, 0, 10, ylim = c(0, 0.1), lwd = 2, xlab = "Time (months)", ylab = "Hazard", 
      cex.axis = 1.5, cex.lab = 1.5)
curve(predhLNT, 0, 37, add=T, lty = 2, lwd = 2)
legend("bottomright", legend = c("Treatment","No Treatment"), lty = c(1,2), lwd = c(2,2),  
       col = c("black","black")) 


## -------------------------------------------------------------------------------------------------------------------------------------------------


# Creating the credible envelopes
tvec <- seq(0,37,by = 0.01)
ntvec <- length(tvec)

hCINT <- matrix(0, ncol = ntvec, nrow = length(ind))
hCIT <- matrix(0, ncol = ntvec, nrow = length(ind))
sCINT <- matrix(0, ncol = ntvec, nrow = length(ind))
sCIT <- matrix(0, ncol = ntvec, nrow = length(ind))

for(j in 1:length(ind)){
  for(k in 1:ntvec){
    lambda <- as.numeric(exp(valx_lNT%*%acoefs[j,]))
    kappa <- as.numeric(exp(valx_kNT%*%bcoefs[j,]))
    hCINT[j,k ] <- hlogisode( tvec[k],lambda, kappa, h0p[j])
    sCINT[j,k ] <- exp(-chlogisode( tvec[k],lambda, kappa, h0p[j]))
  }
  for(k in 1:ntvec){
    lambda <- as.numeric(exp(valx_lT%*%acoefs[j,]))
    kappa <- as.numeric(exp(valx_kT%*%bcoefs[j,]))
    hCIT[j,k ] <- hlogisode( tvec[k],lambda, kappa, h0p[j])
    sCIT[j,k ] <- exp(-chlogisode( tvec[k],lambda, kappa, h0p[j]))
  }
} 

hT <-  hazLT(tvec)
hNT <- hazLNT(tvec)

hCITL <- apply(hCIT, 2, ql)
hCINTL <- apply(hCINT, 2, ql)
hCITU <- apply(hCIT, 2, qu)
hCINTU <- apply(hCINT, 2, qu)


sT <-  survLT(tvec)
sNT <- survLNT(tvec)

sCITL <- apply(sCIT, 2, ql)
sCINTL <- apply(sCINT, 2, ql)
sCITU <- apply(sCIT, 2, qu)
sCINTU <- apply(sCINT, 2, qu)

# Plots

plot(tvec,  hT, type = "l", ylim = c(0, 0.1), xlab = "Time (months)", ylab = "Predictive Hazard", 
       cex.axis = 1.5, cex.lab = 1.5, lwd = 2, lty = 1)
points(tvec,  hNT, col = "black", type = "l", lwd = 2, lty = 2)
points(tvec,  hCITL, col = "gray", type = "l")
points(tvec,  hCITU, col = "gray", type = "l")
points(tvec,  hCINTL, col = "gray", type = "l")
points(tvec,  hCINTU, col = "gray", type = "l")
polygon(c(tvec, rev(tvec)), c(hCITL[order(tvec)], rev(hCITU[order(tvec)])),
        col = "gray", border = NA)
polygon(c(tvec, rev(tvec)), c(hCINTL[order(tvec)], rev(hCINTU[order(tvec)])),
        col = "gray", border = NA)
points(tvec,  hT, type = "l", col = "black", lwd = 2, lty =1)
points(tvec,  hNT, col = "black", type = "l", lwd = 2, lty = 2)
legend("bottomright", legend = c("Treatment", "No Treatment"), lty = c(1,2), 
       lwd = c(2,2), col = c("black","black"))



plot(tvec,  sT, type = "l", ylim = c(0,1), xlab = "Time (months)", ylab = "Predictive Survival", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2, lty = 1)
points(tvec,  sNT, col = "black", type = "l", lwd = 2, lty = 2)
points(tvec,  sCITL, col = "darkgray", type = "l")
points(tvec,  sCITU, col = "darkgray", type = "l")
points(tvec,  sCINTL, col = "gray", type = "l")
points(tvec,  sCINTU, col = "gray", type = "l")
polygon(c(tvec, rev(tvec)), c(sCITL[order(tvec)], rev(sCITU[order(tvec)])),
        col = "darkgray", border = NA)
polygon(c(tvec, rev(tvec)), c(sCINTL[order(tvec)], rev(sCINTU[order(tvec)])),
        col = "gray", border = NA)
points(tvec,  sT, type = "l", col = "black", lwd = 2, lty =1)
points(tvec,  sNT, col = "black", type = "l", lwd = 2, lty = 2)
legend("topright", legend = c("Treatment", "No Treatment"), lty = c(1,2), 
       lwd = c(2,2), col = c("black","black"))



## -------------------------------------------------------------------------------------------------------------------------------------------------


# Creating the credible envelopes
tvec <- seq(0,10,by = 0.01)
ntvec <- length(tvec)

hCINT <- matrix(0, ncol = ntvec, nrow = length(ind))
hCIT <- matrix(0, ncol = ntvec, nrow = length(ind))
sCINT <- matrix(0, ncol = ntvec, nrow = length(ind))
sCIT <- matrix(0, ncol = ntvec, nrow = length(ind))

for(j in 1:length(ind)){
  for(k in 1:ntvec){
    lambda <- as.numeric(exp(valx_lNT%*%acoefs[j,]))
    kappa <- as.numeric(exp(valx_kNT%*%bcoefs[j,]))
    hCINT[j,k ] <- hlogisode( tvec[k],lambda, kappa, h0p[j])
    sCINT[j,k ] <- exp(-chlogisode( tvec[k],lambda, kappa, h0p[j]))
  }
  for(k in 1:ntvec){
    lambda <- as.numeric(exp(valx_lT%*%acoefs[j,]))
    kappa <- as.numeric(exp(valx_kT%*%bcoefs[j,]))
    hCIT[j,k ] <- hlogisode( tvec[k],lambda, kappa, h0p[j])
    sCIT[j,k ] <- exp(-chlogisode( tvec[k],lambda, kappa, h0p[j]))
  }
} 

hT <-  hazLT(tvec)
hNT <- hazLNT(tvec)

hCITL <- apply(hCIT, 2, ql)
hCINTL <- apply(hCINT, 2, ql)
hCITU <- apply(hCIT, 2, qu)
hCINTU <- apply(hCINT, 2, qu)


sT <-  survLT(tvec)
sNT <- survLNT(tvec)

sCITL <- apply(sCIT, 2, ql)
sCINTL <- apply(sCINT, 2, ql)
sCITU <- apply(sCIT, 2, qu)
sCINTU <- apply(sCINT, 2, qu)

# Plots

plot(tvec,  hT, type = "l", ylim = c(0, 0.1), xlab = "Time (months)", ylab = "Predictive Hazard", 
       cex.axis = 1.5, cex.lab = 1.5, lwd = 2, lty = 1)
points(tvec,  hNT, col = "black", type = "l", lwd = 2, lty = 2)
points(tvec,  hCITL, col = "darkgray", type = "l")
points(tvec,  hCITU, col = "darkgray", type = "l")
points(tvec,  hCINTL, col = "gray", type = "l")
points(tvec,  hCINTU, col = "gray", type = "l")
polygon(c(tvec, rev(tvec)), c(hCITL[order(tvec)], rev(hCITU[order(tvec)])),
        col = "darkgray", border = NA)
polygon(c(tvec, rev(tvec)), c(hCINTL[order(tvec)], rev(hCINTU[order(tvec)])),
        col = "gray", border = NA)
points(tvec,  hT, type = "l", col = "black", lwd = 2, lty =1)
points(tvec,  hNT, col = "black", type = "l", lwd = 2, lty = 2)
legend("bottomright", legend = c("Treatment", "No Treatment"), lty = c(1,2), 
       lwd = c(2,2), col = c("black","black"))



plot(tvec,  sT, type = "l", ylim = c(0,1), xlab = "Time (months)", ylab = "Predictive Survival", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2, lty = 1)
points(tvec,  sNT, col = "black", type = "l", lwd = 2, lty = 2)
points(tvec,  sCITL, col = "darkgray", type = "l")
points(tvec,  sCITU, col = "darkgray", type = "l")
points(tvec,  sCINTL, col = "gray", type = "l")
points(tvec,  sCINTU, col = "gray", type = "l")
polygon(c(tvec, rev(tvec)), c(sCITL[order(tvec)], rev(sCITU[order(tvec)])),
        col = "darkgray", border = NA)
polygon(c(tvec, rev(tvec)), c(sCINTL[order(tvec)], rev(sCINTU[order(tvec)])),
        col = "gray", border = NA)
points(tvec,  sT, type = "l", col = "black", lwd = 2, lty =1)
points(tvec,  sNT, col = "black", type = "l", lwd = 2, lty = 2)
legend("topright", legend = c("Treatment", "No Treatment"), lty = c(1,2), 
       lwd = c(2,2), col = c("black","black"))


