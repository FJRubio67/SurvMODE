
load("2A.RData")

library(xtable)

# Summaries
xtable(summr)

# Posteriors

pdf("TrialPost.pdf", width = 16, height = 10)
par(mfrow = c(2,3))
plot(density(alphap0), main = "", xlab = expression(alpha[0]), ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 
abline(v = median(alphap0), lwd = 2, col = "red")
plot(density(alphap1), main = "", xlab = expression(alpha[1]), ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 
abline(v = median(alphap1), lwd = 2, col = "red")
plot(density(betap0), main = "", xlab = expression(beta[0]), ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 
abline(v = median(betap0), lwd = 2, col = "red")
plot(density(betap1), main = "", xlab = expression(beta[1]), ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 
abline(v = median(betap1), lwd = 2, col = "red")
plot(density(h0p), main = "", xlab = expression(h[0]), ylab = "Density", 
     cex.axis = 1.5, cex.lab = 1.5, lwd = 2) 
abline(v = median(h0p), lwd = 2, col = "red")
dev.off()


# Predictive survival Logistic model
pdf("predsurv2A.pdf", width = 8, height = 6)
# Comparison
plot(km, col = c("gray", "gray"), lty = c(2,1), lwd = 2,
     xlab = "Time (months)", ylab = "Predictive Survival", 
     xlim = c(0, 37), ylim = c(0,1), cex.lab = 1.5, cex.axis = 1.5)

curve(predsLT, 0, 37, ylim = c(0,1), xlab = "Time (months)", ylab = "Predictive Survival", lwd = 2, lty = 1, 
      cex.lab = 1.5, cex.axis = 1.5, add = TRUE) 
curve(predsLNT, 0, 37, ylim = c(0,1), lwd = 2, lty = 2, add = TRUE) 
legend("topright", legend = c("Treatment","No Treatment"), lty = c(1,2), lwd = c(2,2),  
       col = c("black","black")) 
dev.off()


# Predictive hazard Logistic model
pdf("predhaz2A.pdf", width = 8, height = 6)
curve(predhLT, 0, 37, ylim = c(0, 0.1), lwd = 2, xlab = "Time (months)", ylab = "Hazard", 
      cex.axis = 1.5, cex.lab = 1.5)
curve(predhLNT, 0, 37, add=T, lty = 2, lwd = 2)
legend("bottomright", legend = c("Treatment","No Treatment"), lty = c(1,2), lwd = c(2,2),  
       col = c("black","black")) 
dev.off()


