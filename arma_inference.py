import numpy as np
import pytensor.tensor as tt
from arviz import summary
from pymc import (HalfNormal, Metropolis, Model, Normal, forestplot,
                   model_to_graphviz, sample)
import pymc as pm
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
    print(phi_means, phi_sd)
    theta_means = model['tab']['params'][-q:]
    theta_sd = model['tab']['bse'][-q:]

    # NaN values can occur in std err (see e.g. stackoverflow.com/questions/35675693 & 210228.
    # We therefore conservatively replace any NaNs by 0.1.
    phi_sd = np.nan_to_num(phi_sd) + np.isnan(phi_sd) * 0.1
    theta_sd = np.nan_to_num(theta_sd) + np.isnan(theta_sd) * 0.1
    m = p + q
    with Model() as model9:
        # alpha = Normal('alpha', mu=0, sigma=10)
        # beta = Normal('beta', mu=0, sigma=10)
        sigma = HalfNormal('sigma', sigma=10)
        if p == 1:
            phi = Normal('phi', mu=phi_means[0], sigma=phi_sd[0])
        else:
            phi = Normal('phi', mu=phi_means, sigma=phi_sd, shape=p)
        if q ==1:
            theta = Normal('theta', mu=theta_means[0], sigma=theta_sd[0])
        else:
            theta = Normal('theta', mu=theta_means, sigma=theta_sd, shape=q)
        y_r = y[m:]
        y = tt.as_tensor(y)
        # x = tt.as_tensor(x)
        # x_r = x[m:]
        resids = y # - beta * x - alpha

        if p == 1:
            u = phi * resids[p - 1: -1]
        else:
            u = tt.add(*[phi[i] * resids[p - (i + 1):-(i + 1)] for i in range(p)])
        eps = resids[p:] - u
        if q == 1:
            v = theta * eps[q - 1: -1]
        else:
            v = tt.add(*[theta[i] * eps[q - (i + 1):-(i + 1)] for i in range(q)])
        mu = u[q:] + v # + alpha + beta * x_r 
        data = Normal('y_r', mu=mu, sigma=sigma, observed=y_r)

    with model9:
        # if q == 1:
        #     step = Metropolis([phi])
        # else:
        #     step = Metropolis([phi, theta])
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=pm.NUTS(), progressbar=False)

    print(summary(trace))
    #plt.show(forestplot(trace, varnames=['alpha', 'beta', 'sigma', 'phi', 'theta']))
    #traceplot(trace, varnames=['alpha', 'beta', 'sigma', 'phi', 'theta'])
    return trace