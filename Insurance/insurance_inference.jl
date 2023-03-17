# set local working directory
wd = pwd()
wd = joinpath(wd, "Insurance")
results_folder = joinpath(wd, "results")

# import libraries
using JLD2
using CSV, DataFrames
using StatsBase
using Tables
# using Impute
using Dates
using Plots, StatsPlots
using PyPlot, PlotlyJS
using GR
using StableRNGs
using MLDataUtils
using Distributions
using DynamicPPL
using Turing
using Random
Random.seed!(111)

# save chains
using HDF5
using MCMCChains
using MCMCChainsStorage

using EvalMetrics # for confusion matrix
using ROCAnalysis
using NamedArrays
# using ROC

inference = false

#### Modelling ####
## (1) load prepared data
@load joinpath(results_folder,"processed insurance data.jld2") X_df_train X_df_test y_train y_test cs
size(X_df_train, 1)

## (2) Bayesian logistic regression (BLR)
# NB: Turing requires data in matrix form, not dataframe
@model function logistic_regression(X_df, y, σ)
    intercept ~ Normal(0, σ)
    coefficients  = Vector(undef, size(X_df, 2))

    for j in 1:length(coefficients)
        coefficients[j] ~ Normal(0, σ) # the order of coefficients is the same as the order of names(X_df)
    end

    for i in 1:size(X_df, 1)
        prob = Distributions.logistic(intercept + ([coefficients[j] * X_df[i, var_name] for (j, var_name) in enumerate(names(X_df))]...))
        # alternatively: using StatsFuns: logistic
        y[i] ~ Bernoulli(prob)
    end
end;

BLR_model = logistic_regression(X_df_train, y_train, 1.0)

## sample from the posterior
if inference == true
    chain = sample(BLR_model, NUTS(0.65), 300, progress=true)
    write(joinpath(results_folder,"insurance_chain_turing.jls"), chain)
else 
    chain = read(joinpath(results_folder,"insurance_chain_turing.jls"), Chains)
end
size(chain)
p = StatsPlots.plot(chain[:, 1:5, 1]) # plot the posterior distribution of parameters
Plots.savefig(p, joinpath(results_folder,"inferred posterior.svg"))

# bivariate (2D) distribution of samples of continuous variables
# selected_parameters = names(chain)[1:4]
# p = MCMCChains.corner(chain, names(chain)[1:4], labels=names(chain)[1:4]) # hopefully we see univariate modal in each pairplot of 2D density.
# Plots.savefig(p, joinpath(results_folder,"inferred posterior bivariate plot.svg"))

## (3) test performance
function predict(X_df, chain, chain_no, threshold)
    X_mat = Matrix(X_df)
    # Pull the means from each parameter's sampled values in the chain.
    intercept = mean(chain[:, :intercept, chain_no])
    coefficient_values  = Vector(undef, size(X_mat, 2))

    for j in 1:length(coefficient_values)
        coefficient_values[j] = mean(chain[:, names(chain)[j+1], chain_no])
    end

    # Retrieve the number of rows.
    n, _ = size(X_mat)

    # Generate a vector to store our predictions.
    probs = Vector{Float64}(undef, n)
    predictions = Vector{Float64}(undef, n)

    # Calculate the logistic function for each element in the test set.
    for i in 1:n
        prob = Distributions.logistic(intercept + ([coefficient_values[j] * X_df[i, var_name] for (j, var_name) in enumerate(names(X_df))]...))
        probs[i] = prob
        if prob >= threshold
            predictions[i] = 1
        else
            predictions[i] = 0
        end
    end
    return probs, Int.(predictions)
end
y_test_preds_prob, y_test_preds = predict(X_df_test, chain, 1, 0.5)
# @save joinpath(results_folder, "y_test_preds_prob.jld2") y_test_preds_prob
write(joinpath(results_folder,"y_test_preds_prob.jls"), y_test_preds_prob)

p=Plots.plot(1:size(y_test, 1), Int.(y_test_preds), color=:blue)
Plots.plot!(1:size(y_test, 1), Int.(y_test), color=:red)
xlabel!(p, "Customer ID")
ylabel!(p, "Fraud or Not")
Plots.savefig(p, joinpath(results_folder,"Test performance: predicted labels.svg"))

cm = EvalMetrics.ConfusionMatrix(y_test, Int.(y_test_preds))
confusion_mat = NamedArray(getproperty.(Ref(cm), [:tn :fn ; :fp :tp]), 
               (["0", "1"], ["0", "1"]), 
               ("pred", "true"))
# alternatively, y_test_preds can be calculated as
labels = y_test_preds.==y_test
mean(labels)
sum(labels)

# ROC curve
any(ismissing.(y_test))
# EvalMetrics.rocplot(y_test, y_test_preds_prob)
# plot the ROC Curve
# roc_data = MLBase.roc(y_test, y_test_preds_prob, [0, 1])
# test_roc = roc_data.roc
# fpr, tpr = roc_data.fprs[2], roc_data.tprs[2]
# plot(fpr, tpr, title="ROC Curve", label="Model",
#      xlabel="False Positive Rate", ylabel="True Positive Rate", legend=:bottomright)
# # Add a diagonal reference line
# plot!([0, 1], [0, 1], color=:black, linestyle=:dash, label="Random")
# p=Plots.plot(test_roc, label="test performance")
# Plots.savefig(p, joinpath(results_folder,"Test performance: ROC.svg"))

# F1 score
f1 = EvalMetrics.f1_score(y_test, y_test_preds)
f1

# Accuracy: note that, accuray doesn't tell the full pic if the data is highly imbalanced.
accuracy = EvalMetrics.accuracy(y_test, y_test_preds)
accuracy

# Precision
precision = EvalMetrics.precision(y_test, y_test_preds)
precision

# Recall
recall = EvalMetrics.recall(y_test, y_test_preds)

# uncertainty estimation for posterior predictives
function predict_withUncertainty(X_df, chain, chain_no, start_idx, batch_size, no_batches, threshold)
    samples_mat = Vector{Any}[]
    for _ in 1:no_batches
        samples_value = mean(chain[rand(start_idx:size(chain, 1), batch_size), :, chain_no]).nt[:mean] # averaging within the batch
        push!(samples_mat, samples_value)
    end
    all_probs = Vector{Any}(undef, no_batches)
    all_predictions = Vector{Any}(undef, no_batches)

    n = size(X_df, 1)
    coefficient_values  = Vector(undef, size(X_df, 2))

    for k in 1:size(samples_mat)[1] # no_batches
        intercept = samples_mat[k][1]
        for j in 1:length(coefficient_values)
            coefficient_values[j] = samples_mat[k][j+1]
        end

        # Generate a vector to store our predictions.
        probs = Vector{Float64}(undef, n)
        predictions = Vector{Float64}(undef, n)

        # Calculate the logistic function for each element in the test set.
        for i in 1:n
            prob = Distributions.logistic(intercept + ([coefficient_values[j] * X_df[i, var_name] for (j, var_name) in enumerate(names(X_df))]...))
            # The prob of a glm model is the probability score of class 1
            probs[i] = prob
            if prob >= threshold
                predictions[i] = 1
            else
                predictions[i] = 0
            end
        end
        all_probs[k] = probs
        all_predictions[k] = predictions
    end
    return all_probs, all_predictions
end

no_batches = 10
all_probs, all_predictions = predict_withUncertainty(X_df_test, chain, 1, 10, 10, no_batches, 0.5)
size(all_probs)
size(all_predictions[1])

# count the frequency of making the two binary predictions
# create a vector to store the probability of making the prediction 1 of out no_batches predictions
uncertainty_vec = Vector{Float64}(undef, size(y_test, 1))
for i in 1:size(y_test,1)
    preds_vec = Int.([all_predictions[k][i] for k in 1:size(all_predictions)[1]])
    prob = sum(preds_vec.==1)/no_batches
    uncertainty_vec[i] =  prob
end
max(uncertainty_vec...)
min(uncertainty_vec...)
size(uncertainty_vec)
p = Plots.plot(1:size(y_test, 1), uncertainty_vec)
Plots.plot!(1:size(y_test, 1), Int.(y_test), color=:red)
xlabel!(p, "Customer ID")
ylabel!(p, "Probabilisty of fraud")
Plots.savefig(p, joinpath(results_folder,"Test performance: predicted probability.svg"))
