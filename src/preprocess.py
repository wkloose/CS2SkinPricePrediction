import glob
import logging
import os
import re
from typing import Optional
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def get_latest_raw_file(raw_dir: str = "data/raw") -> str:
    """Mencari file snapshot raw CSV paling baru berdasarkan waktu modifikasi."""
    raw_files = glob.glob(os.path.join(raw_dir, "cs2_raw_*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"Tidak ada file raw ditemukan di {raw_dir}")
    latest_file = max(raw_files, key=os.path.getmtime)
    logging.info("Menggunakan file raw terbaru: %s", latest_file)
    return latest_file


def clean_currency_string(price_str: Optional[str]) -> float:
    """
    Membersihkan string harga berformat 'Rp 210.000,--' menjadi nilai float.
    """
    if not price_str or pd.isna(price_str):
        return np.nan
    cleaned = re.sub(r"[^\d]", "", str(price_str))
    if not cleaned:
        return np.nan
    return float(cleaned)


def clean_volume_string(volume_str: Optional[str]) -> int:
    """
    Membersihkan string volume seperti '1,250' menjadi integer.
    """
    if not volume_str or pd.isna(volume_str):
        return 0
    cleaned = re.sub(r"[^\d]", "", str(volume_str))
    if not cleaned:
        return 0
    return int(cleaned)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membangun fitur-fitur analitik turunan sesuai spesifikasi LK-03.
    """
    processed = df.copy()

    processed["timestamp"] = pd.to_datetime(processed["timestamp"], utc=True)
    processed["day_of_week"] = processed["timestamp"].dt.dayofweek

    processed["lowest_price_clean"] = processed["lowest_price"].apply(
        clean_currency_string
    )
    processed["median_price_clean"] = processed["median_price"].apply(
        clean_currency_string
    )
    processed["volume_clean"] = processed["volume"].apply(clean_volume_string)

    processed["median_price_clean"] = processed["median_price_clean"].fillna(
        processed["lowest_price_clean"]
    )

    processed = processed.sort_values(by=["market_hash_name", "timestamp"])

    grouped = processed.groupby("market_hash_name")
    processed["price_delta"] = grouped["lowest_price_clean"].diff().fillna(0.0)
    processed["volume_delta"] = grouped["volume_clean"].diff().fillna(0)

    processed["price_ma_3d"] = grouped["lowest_price_clean"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    processed["price_ma_7d"] = grouped["lowest_price_clean"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )

    logging.info("Feature engineering berhasil dikonstruksi.")
    return processed


def run_preprocessing(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed"
) -> str:
    """
    Menjalankan seluruh pipeline pembersihan dan penyimpanan parquet.
    """
    os.makedirs(processed_dir, exist_ok=True)
    latest_raw_file = get_latest_raw_file(raw_dir)

    df_raw = pd.read_csv(latest_raw_file)
    logging.info("Memproses %d baris data mentah...", len(df_raw))

    df_processed = engineer_features(df_raw)

    output_csv = os.path.join(processed_dir, "cs2_market_features.csv")
    output_parquet = os.path.join(processed_dir, "cs2_market_features.parquet")

    df_processed.to_csv(output_csv, index=False)
    df_processed.to_parquet(output_parquet, index=False, engine="pyarrow")

    logging.info("Preprocessing selesai! data bersih tersimpan di:")
    logging.info(" - CSV    : %s (%d baris)", output_csv, len(df_processed))
    logging.info(" - Parquet: %s", output_parquet)
    return output_parquet


if __name__ == "__main__":
    run_preprocessing()
