wd = "/home/yongchao/Desktop/AIUK/Survival/"

# using RDatasets: dataset
using RDatasets
using Survival, StatsModels, CSV, DataFrames, LinearAlgebra, StatsBase
using Turing
using Distributions
using StatsPlots
using DynamicPPL, Random

# Load AML survival data
df = dataset("survival", "kidney")
df
CSV.write(joinpath(wd,"df.csv"), df)

df.eventTime = EventTime.(df.Time, df.Status .== 1)

model = coxph(@formula(eventTime ~ Age + Sex + Disease + Frail), df)

coef(model)
fieldnames(typeof(model))

# extract the mean and standard deviation of the coefficients
mean_coef = coef(model)
std_coef = stderror(model)

# get the names of the coefficients
coef_table = coeftable(model)
fieldnames(typeof(coef_table))
names_coef = coef_table.rownms

p = plot()
for i in 1:length(mean_coef)
    bar!([names_coef[i]], [mean_coef[i]], yerr=[std_coef[i]], label=nothing)
end
xlabel!("Coefficient")
ylabel!("Mean")
title!("Mean and Standard Deviation of Coefficients")

### predict and plot survival curves ###
# define the covariates for the patient of interest
patient_covariates = [28, 1, 1,0,0, 2.3] # disease='GN'

# compute the predicted hazard for the patient of interest
predicted_hazard = exp(dot(coef(model), patient_covariates))


### Implementation using Turing.jl ###
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
        # log_one_minus_risk = log(1 - sum(exp.(log_partial_likelihood)))
        log_one_minus_risk = log(sum(exp.(log_partial_likelihood)))
        
        # include censoring indicator
        if event[i]
            time[i] ~ Exponential(exp(log_partial_likelihood))
        else
            # this is a censored observation
            log_one_minus_risk ~ Exponential(1)
        end
    end
end
# for i in 1:length(λ)
#     event[i] ~ Distributions.Censored(Exponential(λ[i]), time[i], right_censoring=true)
# end

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

# print the summary of the posterior
println(summary(chain))

# plot the trace of the posterior
plot(chain)











# Define predictor variables and response variable
x = Matrix(df[:, ["Age", "Sex", "Time", "Frail"]])
y = df[:, ["Status", "Time"]]



# # Define Cox proportional hazards model
# @model coxph_model(x, y) = begin
#     # Priors
#     β ~ MvNormal(zeros(size(x, 2)), 1)
#     log_sigma ~ Normal(0, 1)
#     sigma = exp(log_sigma)

#     # Linear predictor
#     η = x * β

#     # Likelihood (log-likelihood of Cox proportional hazards model)
#     for i in 1:size(x, 1)
#         logλ_0 ~ Normal(0.5, 0.1)
#         logλ_i = η[i] - log_sigma

#         # Compute partial likelihood
#         risk_set = y[:, 2] .>= y[i, 2]
#         n_risk = sum(risk_set)
#         log_partial_likelihood = logλ_i - log(n_risk) .+ log.(risk_set)

#         # Compute log of one minus risk
#         log_one_minus_risk = log(1 - sum(exp.(log_partial_likelihood))) + logλ_0

#         # Include censoring indicator
#         if y[i, 1] == 1
#             y[i, 2] ~ Exponential(exp(log_partial_likelihood))
#         else
#             # This is a censored observation
#             log_one_minus_risk ~ Exponential(1)
#         end

#     end
# end


# # Compile the model
# model = coxph_model(x, y)

# # Specify number of samples and chains
# n_chains = 4
# n_samples = 1000

# # Sample from the posterior distribution
# chain = sample(model, NUTS(), n_samples, chains=n_chains)

