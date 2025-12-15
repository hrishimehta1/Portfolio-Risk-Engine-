import pytest
from core.data import load_prices_csv, SchemaError

def test_load_prices_ok(tmp_path):
    p = tmp_path / "prices.csv"
    p.write_text("date,ticker,adj_close\n2025-01-02,A,100\n2025-01-03,A,101\n")
    df = load_prices_csv(p)
    assert df["ticker"].nunique() == 1 and "adj_close" in df.columns

def test_load_prices_schema_error(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("d,t,px\n1,2,3\n")
    with pytest.raises(SchemaError):
        _ = load_prices_csv(p)
