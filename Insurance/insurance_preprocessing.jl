# set local working directory
wd = pwd()
wd = joinpath(wd, "Insurance")
results_folder = joinpath(wd, "results")

# import libraries
using JLD2
using CSV, DataFrames, JSON
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


# show(IOContext(stdout, :limit=>false), subtypes(Any))
# helper function
function df2json(df::DataFrame)
    len = length(df[:,1])
    indices = names(df)
    jsonarray = [Dict([string(index) => (ismissing(df[index][i]) ? nothing : df[index][i])
                       for index in indices])
                 for i in 1:len]
    return JSON.json(jsonarray)
end
function save_data_to_json(path::String, df)
    if df isa DataFrame
        open(path,"w") do f
            write(f, df2json(df))
        end
    else
        open(path,"w") do f
            write(f, JSON.json(df))
        end
    end
end

#### 1. Import data ####
data_file_name = "insurance_claims.xls"
data_file_path = joinpath(wd, data_file_name)

df = CSV.File(open(data_file_path)) |> DataFrame
names(df)
foreach(println, names(df)) 

first(df, 6)
size(df)
eltype.(eachcol(df))
show(describe(df), allrows=true, allcols=true)

# examine a particular variable
mean(df[!, :age])
StatsBase.var(df[!, :age])
StatsBase.std(df[:,:age])
describe(df[:, :age]) # also gives any missing counts
summarystats(df.age)

# check class imbalance
print("fraud ratio: ", countmap(df[:,"fraud_reported"])["Y"]/countmap(df[:,"fraud_reported"])["N"] )
# upsampling or downsampling can be applied if needed.

#### 2. Data cleaning ####
## (1) missing value? The are 3 categorical columns having missing values marked as "?": 'collision_type', 'property_damage', 'police_report_available'
# Imputation can be made using `Impute.jl` if needed.
collect(any(ismissing.(c)) for c in eachcol(df))
collect(ismissing(x) for x in eachcol(df))
# vscodedisplay(describe(df, :nmissing)) # count the number of missing cells in each column
mapcols(x -> any(ismissing, x), df)
df = df[:, setdiff(names(df), ["_c39"])] # dropping column "_c39"
names(df)

typeof(missing)
function missing_impute(df, fillNA_value_dict=missing)
    map(names(df)) do col
        if eltype(df[:, col]) <: Number
            total_missing_no = if any(ismissing.(df[:, col])) sum(x->x==missing, df[:, col]) else 0 end
            if total_missing_no > 0
                print("col=$col, total missing no: $total_missing_no")
                if haskey(fillNA_value_dict, col)
                    replace!(df[:, col], missing => fillNA_value_dict[col])
                else
                    replace!(df[:, col], missing => mean(df[:, col]))
                end
            end
        end
        if eltype(df[:, col]) <: AbstractString
            total_missing_no = if any(df[:, col].==convert(eltype(df[:,col]), "?")) sum(x->x==convert(eltype(df[:,col]), "?"), df[:, col]) else 0 end
            if total_missing_no > 0
                print("col=$col, total missing no: $total_missing_no")
                if haskey(fillNA_value_dict, col)
                    df[df[:, col].== convert(eltype(df[:,col]), "?"), col] .= fillNA_value_dict[col]
                else
                    df[df[:, col].== convert(eltype(df[:,col]), "?"), col] .= mode(df[:, col])
                end
            end
        end
    end
    return df
end

fillNA_value_dict = Dict("property_damage"=>"NO", "police_report_available"=>"NO")
df = missing_impute(df, fillNA_value_dict)
df[:, :collision_type]
eltype(df[:, :collision_type])
df[:, :property_damage]
df[:,:police_report_available]

## (2) Drop those columns which are not important for the analysis
df = df[:, setdiff(names(df), ["incident_hour_of_the_day","insured_zip","policy_bind_date","incident_location"])] 
size(df)
[countmap(df[:, col]) for col in names(df)]

## (3) data sanity check
# total claim must be equal to sum of "injury_claim","property_claim","vehicle_claim"
for row in eachrow(df)
    # print("total_claim_amount: $(row["total_claim_amount"])")
    # print("injury_claim+property_claim+vehicle_claim: $(row["injury_claim"]+row["property_claim"] + row["vehicle_claim"])")
    @assert row[:total_claim_amount] == (row[:injury_claim] + row[:property_claim] + row[:vehicle_claim]) "total claims amount doesn't match!!!"
end
# positivity
msgs = []
for row in eachrow(df)
    try
        @assert row[:months_as_customer]>=0
    catch
        push!(msgs, "months_as_customer has negative values")
    end
    try
        @assert row[:age]>=0
    catch
        push!(msgs, "age has negative values")
    end
    try
        @assert row[:policy_annual_premium]>=0
    catch
        push!(msgs, "policy_annual_premium has negative values")
    end
    try
        @assert row[:total_claim_amount]>=0
    catch
        push!(msgs, "total_claim_amount has negative values")
    end
    try
        @assert row[:umbrella_limit]>=0
    catch
        push!(msgs, "umbrella_limit has negative values")
    end
    try
        @assert row[:(capital-gains)]>=0
    catch
        push!(msgs, "capital-gains has negative values")
    end
    try
        @assert row[:number_of_vehicles_involved]>=0
    catch
        push!(msgs, "number_of_vehicles_involved has negative values")
    end
    try
        @assert row[:injury_claim]>=0
    catch
        push!(msgs, "injury_claim has negative values")
    end
    try
        @assert row[:property_claim]>=0
    catch
        push!(msgs, "property_claim has negative values")
    end
    try
        @assert row[:vehicle_claim]>=0
    catch
        push!(msgs, "vehicle_claim has negative values")
    end
end
println(msgs)
# fetch and drop the row with negative umbralla limit
row_idx = findall(df[:, :umbrella_limit].<0)
df = df[Not(row_idx), :]

## (4) datatime coversion and extraction
select(df, r"date")
select(df, eltype.(eachcol(df)) .== Date)

typeof(df[:, "incident_date"])
typeof(df[:, "auto_year"])

df.incident_date = Dates.format.(df.incident_date, dateformat"dd/mm/yyyy") # change the format, output strings
df.incident_date = Dates.Date.(df.incident_date, dateformat"dd/mm/yyyy") # convert it back to Date type
eltype(df.incident_date)
df.auto_year = Dates.Year.(df.auto_year)
eltype(df.auto_year)
eltype.(eachcol(df))

# extract year, month, day, weekday from incident_date
df.incident_year = Int.(Dates.year.(df.incident_date))
df.incident_month = Int.(Dates.month.(df.incident_date))
df.incident_day = Int.(Dates.day.(df.incident_date))
df.incident_weekday = Dates.dayname.(df.incident_date)
df = df[:, setdiff(names(df), ["incident_date", "auto_year"])] # drop the original incident_date column

show(describe(df), allrows=true, allcols=true)

## (5) outlier detection for numeric variables
numeric_cols = names(df)[(<:).(eltype.(eachcol(df)), Union{Number, Missing})]
df[:, numeric_cols]
gr()
for col in numeric_cols
    display(Plots.boxplot(df[:, col], label=col))
end
PlotlyJS.plot(df, x=:insured_sex, y=Symbol(numeric_cols[1]), kind="box")
save_data_to_json(joinpath(results_folder,"numeric_cols.json"), numeric_cols)

## (6) count unique values for categorical variables
Set(df[:, "police_report_available"])
combine(groupby(df, [:insured_sex, :police_report_available]), nrow => :count)

#### 3. EDA ####
## (1) pie plot of fraud ratio
df_count = combine(groupby(df, :fraud_reported), nrow => :count)
PyPlot.pie(df_count[:, "count"], labels=unique(df[:, "fraud_reported"]))
PyPlot.title("fraud_reported ratio")
fig = PyPlot.figure("pyplot_piechart",figsize=(10,10)) # 25% data is reported as fraud
display(fig)

## (2) split whole dataset into fraud and non-fraud sets for density plots use
df_fraud = df[df[:, "fraud_reported"].=="Y", :]
df_non_fraud = df[df[:, "fraud_reported"].=="N", :]

## (3) plot the distribution of numeric variables, as per each fraud/non-fraud class
for col in numeric_cols[1:10]
    # display(StatsPlots.histogram(df_fraud[:, col], label="fraud", alpha=0.5))
    display(StatsPlots.density(df_fraud[:, col]))
    display(StatsPlots.density!(df_non_fraud[:, col]))
end
# we can see the multi-modal nature of some numeric variables.

## (4) boxplot again the claim amount, as per gender / state
PlotlyJS.plot(df, x=:insured_sex, y=:total_claim_amount, kind="box")
PlotlyJS.plot(df, x=:policy_state, y=:total_claim_amount, kind="box")
# @df df StatsPlots.violin(string.(:insured_sex), :total_claim_amount, linewidth=0)
# @df df StatsPlots.boxplot!(string.(:insured_sex), :total_claim_amount, fillalpha=0.75, linewidth=2)
# @df df StatsPlots.dotplot!(string.(:insured_sex), :total_claim_amount, marker=(:black, stroke(0)))

## (5) bivariate distributions
PlotlyJS.plot(PlotlyJS.scatter(x=df[:, "policy_annual_premium"], y=df[:, "total_claim_amount"], mode="markers"))
StatsPlots.cornerplot(Matrix(df[:, ["policy_annual_premium", "total_claim_amount"]]), label = ["policy_annual_premium", "total_claim_amount"])
StatsPlots.corrplot(Matrix(df[:, ["policy_annual_premium", "total_claim_amount"]]), label = ["policy_annual_premium", "total_claim_amount"])

# (6) group by multiple variables 
observe_variables = ["insured_sex", 
"insured_education_level",
"insured_occupation",
"insured_relationship",
"incident_type",
"collision_type",
"incident_severity",
"incident_state",
"property_damage",
"police_report_available",
"witnesses",
"auto_make",
]

# for oberse_var in observe_variables
#     df_groupBy = combine(groupby(df, [Symbol(oberse_var), :fraud_reported]), nrow => :count)
#     display(groupedbar(df_groupBy[:, oberse_var], df_groupBy[:, "count"], group = df_groupBy[:, "fraud_reported"], ylabel = "Number", title = "Number of claims by $oberse_var"))
# end

close("all")

# (7) group densities
p = @df df density(:total_claim_amount, group = (:insured_sex, :insured_education_level), legend = :topright, size = (800, 600))
xlabel!(p, "Total Claim Amount")
ylabel!(p, "Density")
display(p)
Plots.savefig(p, joinpath(results_folder,"female education vs total claim amount.svg"))


# (8) feature correlation heatmap
fig = PyPlot.figure("pariplot",figsize=(600,600))
p = StatsPlots.corrplot(Matrix(df[:, ["months_as_customer","age","policy_annual_premium","total_claim_amount"]]), label = ["months_as_customer","age","policy_annual_premium","total_claim_amount"])
Plots.savefig(p, joinpath(results_folder,"numeric corrplot.svg"))

# (9) pair plot of features
# fig = PyPlot.figure("pariplot",figsize=(10,10)) 
# StatsPlots.pairplot(Matrix(df[:, ["months_as_customer","age","policy_annual_premium","total_claim_amount"]]), label = ["months_as_customer","age","policy_annual_premium","total_claim_amount"])

@save joinpath(results_folder, "df.jld2") df
save_data_to_json(joinpath(results_folder,"df.json"), df)

#### Modelling preparation ####
## separate the target variable from the features
y = [df[i, "fraud_reported"]=="Y" ? 1 : 0 for i in 1:size(df, 1)] # save the target variable before OHE
df_X = df[:, Not("fraud_reported")] # save the features before OHE

## (1) OHE: one hot encoding of categorical variables
cat_cols = names(df_X)[(<:).(eltype.(eachcol(df_X)),Union{AbstractString,Missing})]
function one_hot_encoding(df)
    for (col_name, col) in pairs(eachcol(df)) # col_name is a Symbol
        # if eltype(col) <: Union{AbstractString,Missing}
        if String(col_name) in cat_cols
            print("$col_name \n")
            ux = unique(reduce(vcat, col))
            # @show df[:, col_name]
            transform!(df, col_name .=> [ByRow(v -> (x in [v] ? 1 : 0)) for x in ux] .=> Symbol.("$(String(col_name))_", ux))
            select!(df, Not(col_name))
        end
    end
    return df
end
df_X_ohe= one_hot_encoding(deepcopy(df_X))
@show names(df_X_ohe)
select(df_X_ohe, r"insured_education_level")
first(df_X_ohe, 6)

## (2) shuffle, split, normalise data
df_pre_shuffled = hcat(df_X_ohe, DataFrame(fraud_reported=y))
names(df_pre_shuffled)

rng = StableRNG(111)
df_after_shuffled = shuffleobs(df_pre_shuffled, rng=rng)
"fraud_reported" in names(df_after_shuffled)

df_train, df_test = stratifiedobs(row -> row["fraud_reported"], df_after_shuffled; p=0.8)
names(df_train)

# Turing only accepts matrix as input

response = "fraud_reported"
all_features_ohe= setdiff(names(df_after_shuffled), [response])

# X: the matrix of input samples, of size (d, n). Each column in X is an observation.
X_mat_train_original = Matrix(df_train[:, numeric_cols])
cs = MLDataUtils.fit(FeatureNormalizer, X_mat_train_original')
# Normalizes the given data using the derived parameters
X_mat_train_normalised = (MLDataUtils.predict(cs, X_mat_train_original'))'
X_mat_train_catOHE = Matrix(df_train[:, setdiff(all_features_ohe, numeric_cols)])
X_mat_train = hcat(X_mat_train_normalised, X_mat_train_catOHE)
X_df_train = DataFrame(X_mat_train, Symbol.([numeric_cols..., setdiff(all_features_ohe, numeric_cols)...]))

X_mat_test_original = Matrix(df_test[:, numeric_cols])
X_mat_test_normalised = (MLDataUtils.predict(cs, X_mat_test_original'))'
X_mat_test_catOHE = Matrix(df_test[:, setdiff(all_features_ohe, numeric_cols)])
X_mat_test = hcat(X_mat_test_normalised, X_mat_test_catOHE)
X_df_test = DataFrame(X_mat_test, Symbol.([numeric_cols..., setdiff(all_features_ohe, numeric_cols)...]))

y_train = vec(df_train[:, response])
y_test = vec(df_test[:, response])

## save the train and test DataFrame using JLD2
@save joinpath(results_folder, "processed insurance data.jld2") X_df_train X_df_test y_train y_test cs
save_data_to_json(joinpath(results_folder,"X_df_train.json"), X_df_train)
save_data_to_json(joinpath(results_folder,"X_df_test.json"), X_df_test)
save_data_to_json(joinpath(results_folder,"y_train.json"), y_train)
save_data_to_json(joinpath(results_folder,"y_test.json"), y_test)
save_data_to_json(joinpath(results_folder,"cs.json"), cs)