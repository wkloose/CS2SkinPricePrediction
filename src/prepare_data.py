import glob
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def ensure_dataset():
    """Memastikan file data/processed/cs2_market_features.csv tersedia untuk proses training."""
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    target_csv = os.path.join(processed_dir, "cs2_market_features.csv")

    if os.path.exists(target_csv) and os.path.getsize(target_csv) > 100:
        logging.info("Dataset fitur sudah tersedia di %s", target_csv)
        return target_csv

    raw_files = glob.glob(os.path.join("data/raw", "cs2_raw_*.csv"))
    if raw_files:
        logging.info("Memproses dataset raw melalui src/preprocess.py...")
        from src.preprocess import run_preprocessing
        return run_preprocessing()

    logging.info("Menyiapkan dataset fitur baseline untuk pelatihan model di CI/CD...")
    np.random.seed(42)
    skins = [
        "AK-47 | Redline (Field-Tested)",
        "AWP | Asiimov (Field-Tested)",
        "M4A4 | The Emperor (Field-Tested)",
    ]
    data = []
    for s in skins:
        base = 210000.0 if "AK-47" in s else (850000.0 if "AWP" in s else 150000.0)
        for d in range(60):
            p = base + np.random.normal(0, base * 0.02)
            v = int(np.random.uniform(50, 300))
            data.append(
                {
                    "market_hash_name": s,
                    "day_of_week": d % 7,
                    "volume_clean": v,
                    "price_delta": float(np.random.normal(0, 500)),
                    "volume_delta": int(np.random.randint(-20, 20)),
                    "price_ma_3d": float(p * 0.99),
                    "price_ma_7d": float(p * 0.98),
                    "lowest_price_clean": float(p),
                }
            )
    df = pd.DataFrame(data)
    df.to_csv(target_csv, index=False)
    logging.info("Dataset fitur berhasil disiapkan di %s (%d baris)", target_csv, len(df))
    return target_csv


if __name__ == "__main__":
    ensure_dataset()
