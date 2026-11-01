import os
import sys
import pickle
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestRegressor
from src.train import load_data, preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def export_champion_model(
    data_path: str = "data/processed/cs2_market_features.csv",
    output_dir: str = "models"
):
    """Melatih dan mengekspor model terbaik ke format biner pickle."""
    logging.info("Memuat data olahan dari %s...", data_path)
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, features = preprocess_data(df)

    logging.info("Melatih champion model (Random Forest Tuned Deep)...")
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=16,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": features,
        "model_name": "rf_tuned_deep_20261018"
    }

    target_paths = [os.path.join(output_dir, "model.pkl"), "model.pkl"]
    for path in target_paths:
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logging.info("Model berhasil disimpan ke: %s", path)

    logging.info("Ekspor selesai! jumlah fitur: %d", len(features))

if __name__ == "__main__":
    export_champion_model()
