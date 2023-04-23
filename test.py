import numpy as np
import pytensor.tensor as tt
from statsmodels.tsa.arima_process import arma_generate_sample

# Set sampling params
ndraws = 3000  # number of draws from the distribution
nburn = 600  # number of "burn-in points" (which will be discarded)

from arma_inference import Loglike
import pymc as pm
import pytensor.tensor as tt

np.random.seed(42)
n = 1000
ar_coefs = np.array([.5, .2])
ma_coefs = np.array([-.3, .3])

y = arma_generate_sample(np.r_[1, -ar_coefs], np.r_[1, ma_coefs], nsample=n)
x = np.arange(n)

import statsmodels.api as sm

mod = sm.tsa.statespace.SARIMAX(y, order=(2, 0, 2))

res_mle = mod.fit(disp=False)
print(res_mle.summary())

loglike = Loglike(mod)

with pm.Model() as m:
    # Priors
    arL1 = pm.Uniform("ar.L1", -0.99, 0.99)
    arL2 = pm.Uniform("ar.L2", -0.99, 0.99)
    maL1 = pm.Uniform("ma.L1", -0.99, 0.99)
    maL2 = pm.Uniform("ma.L2", -0.99, 0.99)
    sigma2 = pm.InverseGamma("sigma2", 2, 4)

    # convert variables to tensor vectors
    params = [arL1, arL2, maL1, maL2, sigma2]

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist("likelihood", *params, logp=loglike)

    # Draw samples
    trace = pm.sample(
        ndraws,
        tune=nburn,
        return_inferencedata=True,
        cores=4,
        compute_convergence_checks=False,
    )