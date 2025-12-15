import pandas as pd
import pytest

from core.data import load_prices_csv, SchemaError


def test_load_prices_ok(tmp_path):
    p = tmp_path / "prices.csv"
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01","2024-01-02","2024-01-03"]),
        "ticker": ["AAA","AAA","AAA"],
        "adj_close": [100.0, 101.0, 100.5]
    })
    df.to_csv(p, index=False)
    out = load_prices_csv(p)
    assert set(out.columns) == {"date", "ticker", "adj_close"}
    assert out["ticker"].nunique() == 1


def test_load_prices_bad_schema(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("d,t,px\n1,2,3\n")
    with pytest.raises(SchemaError):
        load_prices_csv(p)
