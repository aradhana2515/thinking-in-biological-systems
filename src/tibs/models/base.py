from abc import ABC, abstractmethod


class Model(ABC):
    """
    Base class for mechanistic biological models.
    """

    @abstractmethod
    def initial_state(self):
        """Return initial state vector."""
        pass

    @abstractmethod
    def parameters(self):
        """Return dict of model parameters."""
        pass

    @abstractmethod
    def stoichiometry(self):
        """Return stoichiometry matrix (N_species x N_reactions)."""
        pass

    @abstractmethod
    def propensities(self, x, t, params):
        """Return reaction propensities for Gillespie."""
        pass

    @abstractmethod
    def rhs(self, t, x, params):
        """ODE right-hand side."""
        pass

    def observe(self, x, params):
        """Map internal state to observable (e.g. luminescence)."""
        return x
