import numpy as np

from tibs.models.gene_expression import GeneExpression


def test_import_and_shapes():
    model = GeneExpression()
    x0 = model.initial_state()
    assert isinstance(x0, np.ndarray)
    assert x0.shape == (2,)

    S = model.stoichiometry()
    assert S.shape == (2, 4)

    p = model.parameters()
    a = model.propensities(x0, t=0.0, p=p)
    assert a.shape == (4,)
    assert np.all(a >= 0)


def test_rhs_dimensions():
    model = GeneExpression()
    x0 = model.initial_state()
    p = model.parameters()
    dx = model.rhs(t=0.0, x=x0, p=p)
    assert dx.shape == (2,)
