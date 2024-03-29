% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/simml-main.R
\name{simml}
\alias{simml}
\title{Single-index models with multiple-links (the main function)}
\usage{
simml(y, Tr, X, mu.hat = NULL, family = "gaussian",
  ortho.constr = TRUE, bs = "ps", k = 8, alpha.ini = NULL,
  ind.to.be.positive = 1, pen.order = 0, lambda = 0, max.iter = 30,
  eps.iter = 0.01, trace.iter = TRUE, bootstrap = FALSE,
  nboot = 200, boot.alpha = 0.05, seed = 1357)
}
\arguments{
\item{y}{a n-by-1 vector of treatment outcomes; y is assumed to follow an exponential family distribution; any distribution supported by \code{mgcv::gam}.}

\item{Tr}{a n-by-1 vector of treatment indicators; each element represents one of the L(>1) treatment conditions; e.g., c(1,2,1,1,1...); can be a factor-valued.}

\item{X}{a n-by-p matrix of pre-treatment covarates.}

\item{mu.hat}{a n-by-1 vector for efficinecy augmentation provided by the user; the defult is \code{NULL}; the optimal choice for this vector is h(E(y|X)), where h is the canonical link function.}

\item{family}{specifies the distribution of y; e.g., "gaussian", "binomial", "poisson"; the defult is "gaussian"; can be any family supported by \code{mgcv::gam}.}

\item{ortho.constr}{separates the interaction effects from the main effect (without this, the interaction effect can be confounded by the main effect; the default is \code{TRUE}.}

\item{bs}{type of basis for representing the treatment-specific smooths; the defult is "ps" (p-splines); any basis supported by \code{mgcv::gam} can be used, e.g., "cr" (cubic regression splines)}

\item{k}{basis dimension; the same number (k) is used for all treatment groups, however, the smooths of different treatments have different roughness parameters.}

\item{alpha.ini}{an initial solution of \code{alpha.coef}; a p-by-1 vector; the defult is \code{NULL}.}

\item{ind.to.be.positive}{for identifiability of the solution \code{alpha.coef}, we restrict the jth component of \code{alpha.coef} to be positive; by default \code{j=1}.}

\item{pen.order}{0 indicates the ridge penalty; 1 indicates the 1st difference penalty; 2 indicates the 2nd difference penalty, used in a penalized least squares (LS) estimation of \code{alpha.coef}.}

\item{lambda}{a regularziation parameter associated with the penalized LS of \code{alpha.coef}.}

\item{max.iter}{an integer specifying the maximum number of iterations for \code{alpha.coef} update.}

\item{eps.iter}{a value specifying the convergence criterion of algorithm.}

\item{trace.iter}{if \code{TRUE}, trace the estimation process and print the differences in \code{alpha.coef}.}

\item{bootstrap}{if \code{TRUE}, compute bootstrap confidence intervals for the single-index coefficients, \code{alpha.coef}; the default is \code{FALSE}.}

\item{nboot}{when \code{bootstrap=TRUE}, a value specifying the number of bootstrap replications.}

\item{boot.alpha}{specifies bootstrap CI percentiles; e.g., 0.05 gives 95\% CIs; 0.1 gives 90\% CIs.}

\item{seed}{when  \code{bootstrap=TRUE}, randomization seed used in bootstrap resampling.}
}
\value{
a list of information of the fitted SIMML including
 \item{alpha.coef}{ the estimated single-index coefficients.} \item{g.fit}{a \code{mgcv:gam} object containing information about the estimated treatment-specific link functions.} \item{alpha.ini}{the initial value used in the estimation of \code{alpha.coef}} \item{alpha.path}{solution path of \code{alpha.coef} over the iterations} \item{d.alpha}{records the change in \code{alpha.coef} over the solution path, \code{alpha.path}} \item{scale.X}{sd of pretreatment covariates X} \item{center.X}{mean of pretreatment covariates X} \item{L}{number of different treatment options} \item{p}{number of pretreatment covariates X} \item{n}{number of subjects} \item{boot.ci}{(1-boot.alpha/2) percentile bootstrap CIs (LB, UB) associated with \code{alpha.coef}}
}
\description{
\code{simml} is the wrapper function for Single-index models with multiple-links (SIMML).
The function estimates a linear combination (a single-index) of covariates X, and models the treatment-specific outcome y, via treatment-specific nonparametrically-defined link functions.
}
\details{
SIMML captures the effect of covariates via a single-index and their interaction with the treatment via nonparametric link functions.
Interaction effects are determined by distinct shapes of the link functions.
The estimated single-index is useful for comparing differential treatment efficacy.
The resulting \code{simml} object can be used to estimate an optimal treatment decision rule
for a new patient with pretreatment clinical information.
}
\examples{

## application of SIMML (on a simulated dataset) (see help(generate.data) for data generation).

family <- "gaussian"   #"poisson"
delta = 1              # moderate main effect
w=2                    # if w=2 (w=1), a nonlinear (linear) contrast function
n=500                  # number of subjects
p=10                   # number of pretreatment covariates

# generate a training dataset
data <- generate.data(n= n, p=p, delta = delta, w= w, family = family)
data$SNR  # the ratio of interactions("signal") vs. main effects("noise") in the canonical param.
Tr <- data$Tr
y  <- data$y
X <- data$X

# generate a (large, 10^5) testing dataset
data.test <- generate.data(n=10^5, p=p, delta = delta,  w= w, family = family)
Tr.test <- data.test$Tr
y.test  <- data.test$y
X.test <- data.test$X
data.test$value.opt     # the optimal "value"


##  estimate SIMML
#1) SIMML without efficiency augmenation
simml.obj1 <- simml(y, Tr, X, family = family)

#2) SIMML with efficiency augmenation
# we can improove efficinecy by using the efficiency augmentation term, mu.hat.
# mu.hat is estimated by a main effect only model (y~ X).
glm.fit <- glm(y ~ X, family=family)  # could also use cv.glmnet to obtain a mu.hat
mu.hat <- as.vector(predict(glm.fit, newx =X, type="link"))
simml.obj2 <- simml(y, Tr, X, mu.hat = mu.hat, family = family)


## apply the estimated SIMMLs to the testing set and obtain treatment assignment rules.
simml.trt.rule1 <- pred.simml(simml.obj1, newx= X.test)$trt.rule
# "value" estimation (estimated by IPWE)
simml.value1 <-  mean(y.test[simml.trt.rule1 == Tr.test])
simml.value1

simml.trt.rule2 <- pred.simml(simml.obj2, newx= X.test)$trt.rule
# "value" estimation (estimated by IPWE)
simml.value2 <-  mean(y.test[simml.trt.rule2 == Tr.test])
simml.value2

# compare these to the optimal "value"
data.test$value.opt


## estimate the MC (modified covariates) model of Tien et al 2014

n.t <- summary(as.factor(Tr)); pi.t <- n.t/sum(n.t)
mc <-  (as.numeric(Tr) + pi.t[1] -2) *cbind(1, X)   # 0.5*(-1)^as.numeric(Tr) *cbind(1, X)
mc.coef  <-  coef(glm(y ~ mc, family =  family))
mc.trt.rule <- (cbind(1, X.test) \%*\% mc.coef[-1] > 0) +1
# "value" estimation (estimated by IPWE)
mc.value  <-  mean(y.test[mc.trt.rule == Tr.test])
mc.value


## visualization of the estimated treatment-specific link functions of SIMML
simml.obj1$alpha.coef      # estimated single-index coefficients
g.fit <- simml.obj1$g.fit   # estimated trt-specific link functions; "g.fit" is a mgcv::gam object.
#plot(g.fit)


## by using the package "mgcViz", we can improve the visualization.
# install.packages("mgcViz")
# mgcViz depends on "rgl". "rgl" depends on XQuartz, which you can download from xquartz.org
# library(mgcViz)

## transform the "mgcv::gam" object to a "mgcViz" object (to improve visualization)
#g.fit <- getViz(g.fit)

#plot1  <- plot( sm(g.fit,1) )  # for treatment group 1
#plot1 + l_fitLine(colour = "red") + l_rug(mapping = aes(x=x, y=y), alpha = 0.8) +
#  l_ciLine(mul = 5, colour = "blue", linetype = 2) + l_points(shape = 19, size = 1, alpha = 0.1) +
#  xlab(expression(paste("z = ", alpha*minute, "x")))  +  ylab("y") +
#  ggtitle("Treatment group 1 (Tr =1)") +  theme_classic()

#plot2 <- plot( sm(g.fit,2) )   # for treatment group 2
#plot2 + l_fitLine(colour = "red") + l_rug(mapping = aes(x=x, y=y), alpha = 0.8) +
#  l_ciLine(mul = 5, colour = "blue", linetype = 2) + l_points(shape = 19, size = 1, alpha = 0.1) +
#  xlab(expression(paste("z = ", alpha*minute, "x"))) +ylab("y") +
#  ggtitle("Treatment group 2 (Tr =2)") + theme_classic()


#trans = function(x) x + g.fit$coefficients[2]

#plotDiff(s1 = sm(g.fit, 2), s2 = sm(g.fit, 1), trans=trans) +  l_ciPoly() +
#  l_fitLine() + geom_hline(yintercept = 0, linetype = 2) +
#  xlab(expression(paste("z = ", alpha*minute, "x")) ) +
#  ylab("(Treatment 2 effect) - (Treatment 1 effect)") +
#  ggtitle("Contrast between two treatment effects") +
#  #geom_vline(xintercept=-0.45, linetype="dotted", color = "red", size=0.8) +
#  theme_classic()


# another way of visualization, using ggplot2
#library(ggplot2)
#dat  <- data.frame(y= simml.obj1$g.fit$model$y,
#                   x= simml.obj1$g.fit$model$single.index,
#                   Treatment= simml.obj1$g.fit$model$Tr)
#g.plot <- ggplot(dat, aes(x=x, y=y, color=Treatment, shape=Treatment, linetype=Treatment)) +
#   geom_point(aes(color=Treatment, shape=Treatment), size=1, fill="white") +
#   scale_colour_brewer(palette="Set1", direction=-1) + theme_classic() +
#   xlab(expression(paste(alpha*minute,"x"))) + ylab("y")
#g.plot + geom_smooth(method=gam, formula= y~ s(x, bs=simml.obj1$bs, k=simml.obj1$k),
#                     se=TRUE, fullrange=TRUE, alpha = 0.35)


#### can obtain bootstrap CIs associated with the single-index coefficients (alpha.coef).
glm.fit <- glm(y ~ X, family=family)  # could also use cv.glmnet.
mu.hat <- as.vector(predict(glm.fit, newx= X, type= "link"))    # efficiency augmentation vector
simml.obj <- simml(y,Tr,X, mu.hat=mu.hat, family=family, bootstrap =TRUE, nboot=15, max.iter=7)
# the default is to use 200 bootstrap replications.
simml.obj$alpha.coef
simml.obj$boot.ci    # displays a set of (1-boot.alpha/2) percentile bootstrap CIs (LB, UB).

# compare the estimates to the true alpha.coef.
data$true.alpha
}
\seealso{
\code{pred.simml},  \code{fit.simml}
}
\author{
Park, Petkova, Tarpey, Ogden
}
