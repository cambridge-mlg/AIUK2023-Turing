using Turing
using FillArrays
using Flux
using Plots
using StatsPlots
using ReverseDiff
using LinearAlgebra
using Random
# Use reverse_diff due to the number of parameters in neural networks.
Turing.setadbackend(:reversediff)

# set local working directory
using Base: joinpath
wd = joinpath("/home/yongchao/Desktop/AIUK", "Insurance")
results_folder = joinpath(wd,"results")
images_folder = joinpath(wd, "images")
# point to Github repo for retrieving data
githubRepo = "https://github.com/YongchaoHuang/AIUK"

# load JUlia pre-processed data
using JSON, DataFrames
X_df_train_str = read(joinpath(results_folder,"X_df_train.json"), String); X_df_train = DataFrame(JSON.parse(X_df_train_str))
X_df_test_str = read(joinpath(results_folder,"X_df_test.json"), String); X_df_test = DataFrame(JSON.parse(X_df_test_str))
y_train_str = read(joinpath(results_folder,"y_train.json"), String); y_train = Vector(JSON.parse(y_train_str))
y_test_str = read(joinpath(results_folder,"y_test.json"), String); y_test = Vector(JSON.parse(y_test_str))

size(X_df_train)
size(X_df_test)

# Construct a neural network using Flux 
nn_initial = Chain(Dense(size(X_df_train,2), 512, tanh), Dense(512, 256, tanh), Dense(256, 1, σ))

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(nn_initial)

length(parameters_initial) # number of paraemters in NN

@model function bayes_nn(xs, ts, nparameters, reconstruct; alpha=0.09)
    # Create the weight and bias vector.
    parameters ~ MvNormal(Zeros(nparameters), I / alpha)

    # Construct NN from parameters
    nn = reconstruct(parameters)
    # Forward NN to make predictions
    preds = nn(xs)

    # Observe each prediction.
    for i in 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end;

# Perform inference.
N = 500
ch = sample(bayes_nn(Matrix(X_df_train)', y_train, length(parameters_initial), reconstruct), MH(), N);
write(joinpath(results_folder,"insurance_chain_turing_BNN.jls"), ch)

p = StatsPlots.plot(ch[:, 1:3, 1]) # plot the posterior distribution of parameters
Plots.savefig(p, joinpath(results_folder,"inferred posterior BNN.svg"))

# Extract all weight and bias parameters.
theta = MCMCChains.group(ch, :parameters).value;

# A helper to create NN from weights `theta` and run it through data `x`
nn_forward(x, theta) = reconstruct(theta)(x)

# calculate the mean value of theta
theta_mean = mean(theta, dims=1)
size(theta_mean)
# prediction using theta_mean
y_test_preds_prob = nn_forward(Matrix(X_df_test)', theta_mean)
write(joinpath(results_folder,"y_test_preds_prob_BNN.jls"), y_test_preds_prob)
@JLD2.save joinpath(results_folder, "y_test_preds_prob_BNN.jld2") y_test_preds_prob

threshold = 0.5
y_test_preds = vec(Int.(y_test_preds_prob .> threshold))

# test accuracy
mean(y_test_preds .== y_test)

# Compute number of true positives, false positives, false negatives, and true negatives
tp = sum(y_test .== 1 .& y_test_preds .== 1)
fp = sum(y_test .== 0 .& y_test_preds .== 1)
fn = sum(y_test .== 1 .& y_test_preds .== 0)
tn = sum(y_test .== 0 .& y_test_preds .== 0)

# Compute accuracy, precision, and F1 score
acc = (tp + tn) / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

# Print results
println("Accuracy: $(round(acc, digits=2))")
println("Precision: $(round(precision, digits=2))")
println("Recall: $(round(recall, digits=2))")
println("F1 score: $(round(f1, digits=2))")
