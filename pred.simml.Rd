% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/simml-main.R
\name{pred.simml}
\alias{pred.simml}
\title{SIMML prediction function}
\usage{
pred.simml(simml.obj, newx, type = "response", maximize = TRUE)
}
\arguments{
\item{simml.obj}{a \code{simml} object}

\item{newx}{a (n-by-p) matrix of new values for the pretreatment covariates X at which predictions are to be made.}

\item{type}{the type of prediction required; the default "response" is on the scale of the response variable;  the alternative "link" is on the scale of the linear predictors.}

\item{maximize}{the default is \code{TRUE}, assuming a larger value of the outcome is better; if \code{FALSE}, a smaller value is assumed to be prefered.}
}
\value{
\item{pred.new}{a (n-by-L) matrix of predicted values; each column represents a treatment option.}
\item{trt.rule}{a (n-by-1) vector of suggested treatment assignments}
}
\description{
This function makes predictions from an estimated SIMML, given a (new) set of pretreatment covariates.
The function returns a set of predicted outcomes for each treatment condition and a set of recommended treatment assignments (assuming a larger value of the outcome is better).
}
\seealso{
\code{simml},\code{fit.simml}
}
\author{
Park, Petkova, Tarpey, Ogden
}
