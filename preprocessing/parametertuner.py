from ABC import ABC, abstractmethod
from .xdog import XDoG
from .fdog import FDoG

class ParameterTuner(ABC):
    @abstractmethod
    def simplify(self):
        pass

    @abstractmethod
    def complicate(self):
        pass

    @abstractmethod
    def return_default(self):
        pass


class XdogTuner(ParameterTuner):
    def __init__(self, xdog: XDoG):
        self.xdog = xdog

        self.default = {
            'epsilon': xdog.epsilon,
            'gamma': xdog.gamma,
            'k': xdog.k,
            'phi': xdog.phi,
            'sigma': xdog.sigma
        }

    def parameter_scale(self, scale):
        self.xdog.epsilon
        self.xdog.gamma
        self.xdog.k
        self.xdog.phi
        self.xdog.sigma

    def simplify(self):
        pass

    def complicate(self):

    def return_default(self):
        self.xdog.epsilon = self.default['epsilon']
        self.xdog.gamma = self.default['gamma']
        self.xdog.k = self.default['k']
        self.xdog.phi = self.default['phi']
        self.xdog.sigma = self.default['sigma']


class FdogTuner(ParameterTuner):
    def __init__(self, fdog: FDoG):
        self.fdog = fdog

        self.default = {
            'sigma_c': fdog.sigma_c,
            'sigma_m': fdog.sigma_m,
            'rho': fdog.rho,
            'tau': fdog.tau
        }

    def parameter_scale(self, scale):
        self.fdog.sigma_c
        self.fdog.sigma_m
        self.fdog.rho
        self.fdog.tau

    def return_default(self):
        self.fdog.sigma_c = self.default['sigma_c']
        self.fdog.sigma_m = self.default['sigma_m']
        self.fdog.rho = self.default['rho']
        self.fdog.tau = self.default['tau']

class AutoParameter(object):
    def __init__(self, tuner: ParameterTuner):
        self.tuner = tuner

    def __call__(self, image):
        pass
