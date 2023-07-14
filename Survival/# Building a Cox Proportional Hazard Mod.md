# Building a Cox Proportional Hazard Model in Julia

This tutorial will guide you through building a Cox proportional hazard model in Julia. We will be using the Survival package to build the model, which is a widely used package for analyzing survival data. The data set we will be using is the kidney data set from the RDatasets package.

## Prerequisites

Before we begin, make sure you have the following packages installed:

- RDatasets
- Survival
- StatsModels
- CSV
- DataFrames
- LinearAlgebra
- StatsBase
- Turing
- Distributions
- StatsPlots
- DynamicPPL
- Random

You can install these packages by running the following command in the Julia REPL:

```julia
using Pkg
Pkg.add(["RDatasets", "Survival", "StatsModels", "CSV", "DataFrames", "LinearAlgebra", "StatsBase", "Turing", "Distributions", "StatsPlots", "DynamicPPL", "Random"])
```

## Importing and Preprocessing Data
In this tutorial, we will be using the kidney data set from the RDatasets package. To import this data set, we will use the dataset function from the RDatasets package. We will then preprocess the data set by creating a new column eventTime that represents the time to event.

```julia
wd = "/home/yongchao/Desktop/AIUK/Survival/"

# import required packages
using RDatasets
using Survival, StatsModels, CSV, DataFrames, LinearAlgebra, StatsBase
using Turing
using Distributions
using StatsPlots
using DynamicPPL, Random

# load data set
df = dataset("survival", "kidney")
df

# create new column for event time
df.eventTime = EventTime.(df.Time, df.Status .== 1)

# save the preprocessed data as a csv file
CSV.write(joinpath(wd,"df.csv"), df)

```

## Building the Cox Proportional Hazard Model
Now that we have preprocessed our data, we can start building the Cox proportional hazard model. To build this model, we will use the coxph function from the Survival package. We will specify the formula for the model using the @formula macro.

```julia
# build Cox proportional hazard model
model = coxph(@formula(eventTime ~ Age + Sex + Disease + Frail), df)

# print out the coefficients of the model
coef(model)

# print out the field names of the model object
fieldnames(typeof(model))

```

## ANOVA
We can extract the mean and standard deviation of the coefficients using the coef and stderror functions from the Survival package.

```julia
# extract the mean and standard deviation of the coefficients
mean_coef = coef(model)
std_coef = stderror(model)
```

To visualize the mean and standard deviation of the coefficients, we will create a bar plot using the StatsPlots package.

```julia
# get the names of the coefficients
coef_table = coeftable(model)
fieldnames(typeof(coef_table))
names_coef = coef_table.rownms

# create bar plot of coefficients
p = plot()
for i in 1:length(mean_coef)
    bar!([names_coef[i]], [mean_coef[i]], yerr=[std_coef[i]], label=nothing)
end
xlabel!("Coefficient")
ylabel!("Mean")
title!("Mean and Standard Deviation of Coefficients")

```

## Kaplan-Meier Survival Curve

The Kaplan-Meier survival curve shows the probability of surviving past a certain time, given that the individual has survived up to that time. We can create this plot using the Survival.jl package.

```julia
using Plots
using Survival

# prepare data
t = df.Time
status = df.Status

# fit the Kaplan-Meier survival curve
km = fit(KaplanMeier, t, status)

# plot the survival curve
plot(km, legend=false, xlabel="Time", ylabel="Probability of Survival")

```


## Building the Cox Proportional Hazard Model using Turing.jl

In this section, we will build the Cox proportional hazard model using the Turing.jl package. We will define the model using the @model macro and specify the priors on the coefficients. We will then define the hazard function and the likelihood using the partial likelihood method.

```julia
# import required packages
using Turing
import Base: exp
using Distributions
import Distributions: Censored

# define the model
@model function coxph_turing(time, event, age, sex, disease, frailty)
    # priors on the coefficients
    α ~ Normal()
    β_age ~ Normal()
    β_sex ~ Normal()
    β_disease_GN ~ Normal()
    β_disease_AN ~ Normal()
    β_disease_PKD ~ Normal()
    β_disease_Other ~ Normal()
    β_frailty ~ Normal()
    
    # hazard function
    λ = exp.(fill(α, length(age)) + β_age * age + β_sex * sex +
            β_disease_GN * (disease .== "GN") +
            β_disease_AN * (disease .== "AN") +
            β_disease_PKD * (disease .== "PKD") +
            β_disease_Other * (disease .== "Other") +
            β_frailty * frailty)
    
    # likelihood
    for i in 1:length(time)
        # compute the partial likelihood
        risk_set = time .>= time[i]
        n_risk = sum(risk_set)
        log_partial_likelihood = log(λ[i]) - log(n_risk)
        
        # compute log of one minus risk
        log_one_minus_risk = log(sum(exp.(log_partial_likelihood))+esp())
        
        # include censoring indicator
        if event[i]
            time[i] ~ Exponential(exp(log_partial_likelihood))
        else
            # this is a censored observation
            log_one_minus_risk ~ Exponential(1)
        end
    end
end

```

We will then prepare the data and define the sampler to sample the posterior.

```julia
# prepare data
time = df.Time
event = df.Status .== 1
age = df.Age
sex = df.Sex .== 1
disease = String.(df.Disease)
frailty = df.Frail

# define the sampler and sample the posterior
n_samples = 1000
chain = sample(coxph_turing(time, event, age, sex, disease, frailty), NUTS(), n_samples)

```

We can visualize the trace of the posterior for each parameter in the Cox proportional hazard model using a trace plot. We can create this plot using the Turing.jl package.

```julia
using Turing

# define the sampler and sample the posterior
n_samples = 1000
chain = sample(coxph_turing(time, event, age, sex, disease, frailty), NUTS(), n_samples)

# plot the trace of the posterior
plot(chain)

```


Finally, we will print the summary of the posterior and plot the trace of the posterior.

```julia 
# print the summary of the posterior
println(summary(chain))

# plot the trace of the posterior
plot(chain)

```

Note that we used the partial likelihood method to compute the likelihood. We first computed the partial likelihood for each observation, which is the likelihood of the event occurring at that time given the hazard function and the observations up to that time. We then computed the log of one minus the risk, which is the likelihood of the event not occurring up to that time given the hazard function and the observations up to that time. We included a censoring indicator to handle censored observations, which are observations where the event did not occur by the end of the study.

