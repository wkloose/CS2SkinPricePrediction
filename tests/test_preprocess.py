import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from src.preprocess import (
    clean_currency_string,
    clean_volume_string,
    engineer_features,
)


def test_clean_currency_string_valid():
    """Menguji pembersihan string harga IDR menjadi format float."""
    assert clean_currency_string("Rp 210.000,--") == 210000.0
    assert clean_currency_string("Rp 15.500") == 15500.0
    assert clean_currency_string("1250000") == 1250000.0


def test_clean_currency_string_invalid():
    """Menguji penanganan input harga kosong atau bernilai null."""
    assert np.isnan(clean_currency_string(None))
    assert np.isnan(clean_currency_string(""))
    assert np.isnan(clean_currency_string(np.nan))


def test_clean_volume_string_valid():
    """Menguji konversi string volume ribuan menjadi integer."""
    assert clean_volume_string("1,250") == 1250
    assert clean_volume_string("50") == 50
    assert clean_volume_string("0") == 0


def test_clean_volume_string_invalid():
    """Menguji penanganan input volume null menghasilkan default 0."""
    assert clean_volume_string(None) == 0
    assert clean_volume_string("") == 0
    assert clean_volume_string(np.nan) == 0


def test_engineer_features_structure():
    """Menguji apakah fungsi feature engineering menghasilkan kolom-kolom analitik yang valid."""
    mock_data = pd.DataFrame(
        {
            "market_hash_name": ["AK-47 | Redline (Field-Tested)"] * 3,
            "timestamp": [
                "2026-10-01 10:00:00",
                "2026-10-02 10:00:00",
                "2026-10-03 10:00:00",
            ],
            "lowest_price": ["Rp 200.000,--", "Rp 210.000,--", "Rp 220.000,--"],
            "median_price": ["Rp 205.000,--", "Rp 215.000,--", "Rp 225.000,--"],
            "volume": ["100", "150", "200"],
        }
    )

    df_result = engineer_features(mock_data)

    expected_columns = [
        "lowest_price_clean",
        "median_price_clean",
        "volume_clean",
        "price_delta",
        "volume_delta",
        "price_ma_3d",
        "price_ma_7d",
        "day_of_week",
    ]
    for col in expected_columns:
        assert col in df_result.columns, f"Kolom {col} tidak ditemukan!"

    assert df_result["lowest_price_clean"].iloc[0] == 200000.0
    assert df_result["lowest_price_clean"].iloc[1] == 210000.0
    assert df_result["price_delta"].iloc[1] == 10000.0
    assert df_result["volume_clean"].iloc[2] == 200
