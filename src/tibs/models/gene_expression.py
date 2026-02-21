import numpy as np

from .base import Model


class GeneExpression(Model):
    """
    Simple transcription–translation model:
    DNA -> mRNA -> Protein
    """

    def __init__(self, k_tx=1.0, k_tl=5.0, gamma_m=0.2, gamma_p=0.05):
        self.params = dict(k_tx=k_tx, k_tl=k_tl, gamma_m=gamma_m, gamma_p=gamma_p)

    def initial_state(self):
        # [mRNA, protein]
        return np.array([0, 0], dtype=float)

    def parameters(self):
        return self.params

    def stoichiometry(self):
        # reactions:
        # 0: transcription
        # 1: mRNA decay
        # 2: translation
        # 3: protein decay
        return np.array(
            [
                [+1, -1, 0, 0],  # mRNA
                [0, 0, +1, -1],  # protein
            ]
        )

    def propensities(self, x, t, p):
        m, prot = x
        return np.array(
            [
                p["k_tx"],
                p["gamma_m"] * m,
                p["k_tl"] * m,
                p["gamma_p"] * prot,
            ]
        )

    def rhs(self, t, x, p):
        m, prot = x
        dm = p["k_tx"] - p["gamma_m"] * m
        dp = p["k_tl"] * m - p["gamma_p"] * prot
        return np.array([dm, dp])

    def observe(self, x, params):
        # luminescence proxy
        return x[1]
