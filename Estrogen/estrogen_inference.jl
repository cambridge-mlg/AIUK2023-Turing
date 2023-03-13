# ref: https://www.oxinabox.net/2022/11/11/Estimating-Estrogen.html
using Plots

function single_dose(c_max, halflife, t_max)
    function(t)
        if t < t_max
            c_max/t_max * t
        else
            c_max * 2^(-(t-t_max)/halflife)
        end
    end
end


## simulation
using Plots
plot(layout=(1,3))

plot!(
    0:0.1:24, single_dose(132, 3.5, 3),
    label="A200 predicted"; linecolor=:red, subplot=1, legend=:topright
)
scatter!(
    [0,1,2,3,4,6,8,10,12,16,24], [0,25,100,132,90,82,60,55,32,15,4],
    label="A200 actual", markercolor=:red, subplot=1, legend=:topright
); ylims!(0, 130)

plot!(
    single_dose(100, 3.5, 2.5),
    label="A400 predicted", linecolor=:magenta, subplot=2, legend=:topright
)
scatter!(
    [0,1,2,3,4,6,8,10,12,16,24], [25,35,70,75,55,45,35,32,22,15,4];
    label="A400 actual", markercolor=:magenta, subplot=2, legend=:topright
); ylims!(0, 130)

plot!(
    single_dose(20, 3.5, 2.7),
    label="Amax predicted", linecolor=:blue, subplot=3, legend=:topright
)
scatter!(
    [0,1,2,3,4,6,8,10,12,16,24], [5,14,17,20,12,10,5,2,5,5,4];
    label="Amax actual", markercolor=:blue, subplot=3, legend=:topright
); ylims!(0, 130)

# save figure to /home/yongchao/Desktop/new\ julia/UK-AI/Estrogen/
savefig("/home/yongchao/Desktop/new julia/UK-AI/Estrogen/simulated.svg")

## least squares fitting
using Optim
using Statistics

t = [0,1,2,3,4,6,8,10,12,16,24]
y = [0,25,100,132,90,82,60,55,32,15,4]
p0 = [132, 3.5, 3] #(c_max, halflife, t_max)

function least_squares(f, x, y, p0)
    function loss(p)
        sum((f(p...).(x) - y).^2)
    end
    Optim.optimize(loss, p0)
end

c_max, halflife, t_max = least_squares(single_dose, t, y, p0).minimizer

plot(
    0:0.1:24, single_dose(c_max, halflife, t_max),
    label="A200 predicted"; linecolor=:red,
)
scatter!(
    x, y,
    label="A200 actual", markercolor=:red,
)

savefig("/home/yongchao/Desktop/new julia/UK-AI/Estrogen/least_squares_fit.svg")

## probabilistic programming
using Turing, StatsPlots, Distributions

@model function single_dose_model(t, y)
    c_max ~ Uniform(0, 500)
    # halflife and t_max can be truncated Normal
    halflife ~ Truncated(Normal(3, 1), 0, Inf)
    t_max ~ Truncated(Normal(3, 1), 0, Inf)
    err ~ Gamma(1, 1)

    dose_f = single_dose(c_max, halflife, t_max)
    dose_f_arr = dose_f.(t)
    y ~ MvNormal(dose_f_arr, err)
end

t = [1,2,3,4,6,8,10,12,16,24]
y = [25,100,132,90,82,60,55,32,15,4]

model = single_dose_model(t, y)
chain = sample(model, NUTS(), 4000)
plot(chain)
# save chain
write("estrogen_chain.jls", chain)
savefig("/home/yongchao/Desktop/new julia/UK-AI/Estrogen/estrogen_inference_results.svg")

## posterior predictive
# chain = read("estrogen_chain.jls", JLSOFile)
using Tables


gr(fmt=:png)  # make sure not to plot svg or will crash browser

t = [0,1,2,3,4,6,8,10,12,16,24]
y = [25,35,70,75,55,45,35,32,22,15,4]

scatter(t, y, label="A200 actual", markercolor=:red, legend=false)

for samp in rowtable(chain)
    f = single_dose(samp.c_max, samp.halflife, samp.t_max)
    plot!(0:0.1:24, f, linewidth=0.5, linealpha=0.005, linecolor=:red)
end

# Display the plot with all the curves
display(plot!)
savefig("/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_allData.svg")

## condition on a single data point
model = single_dose_model([8],[60])
chain=sample(model, NUTS(), 4_000)

scatter(
    [8], [60],
    label="A200 actual", markercolor=:red,
    legend=false
)

for samp in rowtable(chain)
    f = single_dose(samp.c_max, samp.halflife, samp.t_max)
    plot!(
        0:0.1:24, f,
        linewidth=1,
        linealpha=0.005, linecolor=:red,
    )
end
display(plot!)
savefig("/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_singleData.svg")

## condition on two data points
model = single_dose_model([3,8], [100,60]) | (;c3=100, c8=60)
chain=sample(model, NUTS(), 4_000)

scatter(
    [3, 8], [100, 60],
    label="A200 actual", markercolor=:red,
    legend=false
)

for samp in rowtable(chain)
    f = single_dose(samp.c_max, samp.halflife, samp.t_max)
    plot!(
        0:0.1:24, f,
        linewidth=1,
        linealpha=0.005, linecolor=:red,
    )
end
display(plot!)
savefig("/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_twoData.svg")

# three data points
model = single_dose_model([1,3,8], [50,100,60])
chain=sample(model, NUTS(), 4_000)

scatter(
    [3, 8, 1], [100, 60, 50],
    label="A200 actual", markercolor=:red,
    legend=false
)

for samp in rowtable(chain)
    f = single_dose(samp.c_max, samp.halflife, samp.t_max)
    plot!(
        0:0.1:24, f,
        linewidth=1,
        linealpha=0.005, linecolor=:red,
    )
end
display(plot!)
savefig("/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_threeData1.svg")

# three data points
model = single_dose_model([1,2,3], [30,90,90])
chain=sample(model, NUTS(), 4_000)

scatter(
    [3, 8, 1], [100, 60, 50],
    label="A200 actual", markercolor=:red,
    legend=false
)

for samp in rowtable(chain)
    f = single_dose(samp.c_max, samp.halflife, samp.t_max)
    plot!(
        0:0.1:24, f,
        linewidth=1,
        linealpha=0.005, linecolor=:red,
    )
end
display(plot!)
savefig("/home/yongchao/Desktop/new julia/UK-AI/Estrogen/posterior_predictive_threeData2.svg")