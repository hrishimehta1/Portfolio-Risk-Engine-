import pandas as pd
from core.model import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio


def test_metrics_shapes():
    s = pd.Series([0.01, 0.00, -0.01, 0.02, -0.005])
    assert isinstance(sharpe_ratio(s), float)
    assert isinstance(sortino_ratio(s), float)
    assert max_drawdown(s) <= 0.0
    assert isinstance(calmar_ratio(s), float)
