# set local working directory
import os
cwd = os.getcwd() # localPath = "/home/yongchao/Desktop/AIUK"
results_folder = cwd + "/results"

# choose which inference engine to use: PyMC3 or Turing
inference_engine = "Turing"

#### Model and Inference ####
if inference_engine == "PyMC3":
    # import libraries
    import numpy as np
    import pymc3 as pm
    ### (1) Define the 2-RC Equivalent Circuit Model for battery. ###
    class towRCs_ECM:
        def __init__(self, rt, r1_, t1, r2_, t2, omega):
            """
            Args:
                - Rt: the total resistance of the battery (aka R at f=0 [Hz])
                - rt: the normalised DC resistance
                - ri: the normalised AC resistance of the i-th RC pair
                - ti: the normalised time constant of the i-th RC pair
                - sigma: experimental noise variance
                - omega: angular frequencies [rad/s]
            """
            self.omega = omega
            self.omega_mu = np.mean(np.log(self.omega))
            self.omega_sigma = np.std(np.log(self.omega))
            
            self.rt = rt
            self.t1 = t1
            self.r1 = np.exp(-np.exp(r1_))
            self.t2 = t2
            self.r2 = np.exp(-np.exp(r2_))
            self.Rt = np.exp(self.rt)
            self.r0 = 1 - self.r1 - self.r2
    
        def real_part(self):
            """
            Returns: real part of impedance spectrum
            """
            return self.Rt * (
                self.r0 + self.r1 / 2 * (1 - np.tanh(np.log(self.omega) - (self.omega_sigma * self.t1 + self.omega_mu)))
                + self.r2 / 2 * (1 - np.tanh(np.log(self.omega) - (self.omega_sigma * self.t2 + self.omega_mu)))
            )

        def imarginary_part(self):
            """
            Returns: imaginary part of impedance spectrum
            """
            return self.Rt * (
                (self.r1 / 2) / np.cosh(np.log(self.omega) - (self.omega_sigma * self.t1 + self.omega_mu))
                + (self.r2 / 2) / np.cosh(np.log(self.omega) - (self.omega_sigma * self.t2 + self.omega_mu))
            )

    # instantiate the battery model, and generate simulated gdata.
    n = 100
    f = np.logspace(1, 10, n)  # frequency [Hz]
    angular_f = 2 * np.pi * f
    rt, r1, t1, r2, t2, noise_std = [2, -0.5, -1, 0, 0.5, 0.01]
    twoRCs_model_obj = towRCs_ECM(rt, r1, t1, r2, t2, omega=angular_f)

    noise = np.random.normal(0, noise_std, n)
    reals_simulated = twoRCs_model_obj.real_part() + noise
    ims_simulated = twoRCs_model_obj.imarginary_part() +noise

    log_angular_f = np.log10(twoRCs_model_obj.omega / (2 * np.pi))
    
    ### (2) inference of the ECM parameters using PyMC3 ###
    import pickle
    import pymc3 as pm; print(pm.__version__)    
    
    with pm.Model() as pm_model:
        # define priors on each param
        rt = pm.Uniform("rt", lower=0, upper=10, transform=None)
        r1 = pm.Uniform("r1", lower=-5, upper=5, transform=None)
        t1 = pm.Uniform("t1", lower=-5, upper=5, transform=None)
        r2 = pm.Uniform("r2", lower=-5, upper=5, transform=None)
        t2 = pm.Uniform("t2", lower=-5, upper=5, transform=None)
        noise_std = pm.Uniform("noise_std", lower=0, upper=1, transform=None)

        TwoRCsModel = towRCs_ECM(rt, r1, t1, r2, t2, omega=angular_f) # omega are the constant angular frequencies.
        reals_sol = TwoRCsModel.real_part()
        ims_sol = TwoRCsModel.imarginary_part()
        sol = pm.theano.tensor.stack([reals_sol, ims_sol]).T

        # likelihoods
        obs = pm.theano.tensor.stack((reals_simulated, ims_simulated)).T
        Y_obs = pm.Normal("Y_obs", mu=sol, sigma=noise_std, observed=obs) # NB: noise sigma has been embedded into sol from the TwoRCsModel.
        trace = pm.sample(step=pm.NUTS(target_accept=0.8), draws=1000, tune=500, chains=1, cores=2, return_inferencedata=False, progressbar=True)
    # save the model
    # pm.save_model(pm_model, "/home/yongchao/Desktop/UK AI/battery_model.pkl")
    with open(results_folder+"/battery_model_pymc.pkl", "wb") as f:
        pickle.dump(pm_model, f)
    # save the trace
    pm.save_trace(trace, directory=results_folder+"/battery_trace_pymc", overwrite=True)

if inference_engine == "Turing":
    os.chdir(results_folder)
    from julia.api import Julia
    j = Julia(compiled_modules=False)
    j.eval("""using Pkg; 
            Pkg.activate(); Pkg.instantiate()
            Pkg.add("Turing")
            Pkg.add("Random")
            Pkg.add("Statistics")

            using Turing, Random, Statistics

            ### (1) Define theECM  mode: the function to calculate the impedance spectra ###
            function TwoRCs_ECM(rt::Real, r1::Real, t1::Real, r2::Real, t2::Real, omega::Vector{<:Real})
                omega_mu = mean(log.(omega))
                omega_sigma = std(log.(omega))
                
                R_t = exp(rt)
                r1_ = exp(-exp(r1))
                r2_ = exp(-exp(r2))
                r0 = 1 - r1_ - r2_
                
                real_part = (r0 .+ r1_/2 * (1 .- tanh.(log.(omega) .- (omega_sigma * t1 + omega_mu))) .+
                        r2_/2 * (1 .- tanh.(log.(omega) .- (omega_sigma * t2 + omega_mu)))) .* R_t

                imaginary_part = ((r1_/2) ./ cosh.(log.(omega) .- (omega_sigma * t1 + omega_mu)) .+
                        (r2_/2) ./ cosh.(log.(omega) .- (omega_sigma * t2 + omega_mu))) .* R_t

                return real_part, imaginary_part
            end

            # instantiate the battery model, and generate simulated data.
            n = 100
            f = 10 .^(range(1, stop=10, length=n))  # frequency [Hz]
            angular_f = 2 .* pi .* f
            rt, r1, t1, r2, t2, noise_std = [2, -0.5, -1, 0, 0.5, 0.01]
            reals_simulated, ims_simulated = TwoRCs_ECM(rt, r1, t1, r2, t2, angular_f)

            Random.seed!(123)
            noise = randn(n) .* noise_std
            reals_simulated = reals_simulated .+ noise
            ims_simulated = ims_simulated .+ noise

            log_angular_f = log10.(angular_f ./ (2 .* pi))

            ### (2) Bayesian inference of the battery model using Turing.jl ###
            @model function TwoRCs_ECM_model(reals_simulated, ims_simulated, log_angular_f)
                # define priors on each param
                rt ~ Uniform(0, 10)
                r1 ~ Uniform(-5, 5)
                t1 ~ Uniform(-5, 5)
                r2 ~ Uniform(-5, 5)
                t2 ~ Uniform(-5, 5)
                noise_std ~ Uniform(0, 1)

                # calculate the impedance spectra
                reals_sol, ims_sol = TwoRCs_ECM(rt, r1, t1, r2, t2, angular_f)

                # likelihoods
                reals_simulated ~ MvNormal(reals_sol, noise_std)
                ims_simulated ~ MvNormal(ims_sol, noise_std)
            end

            # instantiate the model
            model = TwoRCs_ECM_model(reals_simulated, ims_simulated, log_angular_f)

            # sample from the posterior
            chain = sample(model, NUTS(), 1000)

            # save the trace
            write("battery_chain_turing.jls", chain)
    """)