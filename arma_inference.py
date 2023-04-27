import numpy as np
import pytensor.tensor as tt
from arviz import summary
from pymc import (HalfNormal, Metropolis, Model, Normal, forestplot,
                   model_to_graphviz, sample)

import pytensor.tensor as tt

class Loglike(tt.Op):
    itypes = [tt.dvector] # parameter vector
    otypes = [tt.dscalar] # log-likelihood scalar

    def __init__(self, model):
        self.model = model
        self.score = Score(self.model) 
    
    def perform(self, node, inputs, outputs):
        theta, = inputs  # contains the vector of parameters
        llf = self.model.loglike(theta)
        outputs[0][0] = np.array(llf)  # output the log-likelihood

    def grad(self, inputs, g):
        return [self.score(inputs[0])]
    

class Score(tt.Op):
    itypes = [tt.dvector] # parameter vector
    otypes = [tt.dvector] # score vector

    def __init__(self, model):
        self.model = model

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        outputs[0][0] = self.model.score(theta)


def run_MCMC_ARMApq(y, draws, model):
    """Derive slope and intercept for ARMA(p, q) model with known p nd q.
    We initially fit a model to the residuals using statsmodels.tsa.api.ARMA.
    Details of this model (as produced by ARMA_select_models.py) are provided as a parameter
    to the present function to allow derivation of reasonably accurate prior distributions for phi and theta.
    If these priors are too broad, the MCMC will not converge in a reasonable time."""

    p = model['order'][0]
    q = model['order'][1]
    phi_means = model['tab']['params'][:p]
    phi_sd = model['tab']['bse'][:p]

    theta_means = model['tab']['params'][-q:]
    theta_sd = model['tab']['bse'][-q:]

    # NaN values can occur in std err (see e.g. stackoverflow.com/questions/35675693 & 210228.
    # We therefore conservatively replace any NaNs by 0.1.
    phi_sd = np.nan_to_num(phi_sd) + np.isnan(phi_sd) * 0.1
    theta_sd = np.nan_to_num(theta_sd) + np.isnan(theta_sd) * 0.1
    m = p + q
    if len(y) < m:
        raise ValueError('Data must be at least as long as the sum of p and q.')
    
    with Model() as model9:
        # alpha = Normal('alpha', mu=0, sigma=10)
        # beta = Normal('beta', mu=0, sigma=10)
        sigma = HalfNormal('sigma', sigma=1)
        phi = Normal('phi', mu=phi_means, sigma=phi_sd, shape=p)
        theta = Normal('theta', mu=theta_means, sigma=theta_sd, shape=q)
            
        y_eff = y[m:]
        l = len(y)
        y = tt.as_tensor(y) 
        residuals = tt.zeros_like(y)
        for t in range(l):
            ar_terms = tt.dot(phi, y[t - p:t][::-1]) if t >= p else tt.zeros_like(y[t])
            ma_terms = tt.dot(theta, residuals[t - q:t][::-1]) if t >= q else tt.zeros_like(y[t])
            residuals = tt.set_subtensor(residuals[t], y[t] - ar_terms - ma_terms)

        mu = tt.add(*[phi[i] * y[p - (i + 1):-(i + 1)] for i in range(p)])[q:] + tt.add(*[theta[i] * residuals[q-i-1 : -i-1] for i in range(q)])[p:] + residuals[m:]

        likelihood = Normal('y_r', mu=mu, sigma=sigma, observed=y_eff)

    with model9:
        if q == 1:
            step = Metropolis([phi]) # use metropolis for phi
        else:
            step = Metropolis([phi, theta])
        tune = draws // 2
        trace = sample(draws, step=step, tune=tune, progressbar=True, chains=4, cores=-1)

    print(summary(trace, var_names=['phi', 'theta', 'sigma']))
    return trace