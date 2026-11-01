import os
import pickle
import logging
import numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)

MODEL_PATH = "models/model.pkl" if os.path.exists("models/model.pkl") else "model.pkl"

logging.info("Memuat model dari: %s...", MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    saved_payload = pickle.load(f)
    if isinstance(saved_payload, dict) and "model" in saved_payload:
        model = saved_payload["model"]
        feature_names = saved_payload.get("feature_names", [])
        model_name = saved_payload.get("model_name", "rf_tuned_deep_20261018")
    else:
        model = saved_payload
        feature_names = []
        model_name = "rf_tuned_deep_20261018"

logging.info("Model '%s' berhasil dimuat. total fitur yang diharapkan: %d", model_name, len(feature_names))

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    """Endpoint status kesehatan layanan inferensi."""
    return jsonify({
        "status": "healthy",
        "service": "CS2 Skin Price Prediction API",
        "model": model_name,
        "features_count": len(feature_names)
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint inferensi prediksi harga skin CS2."""
    try:
        content = request.get_json(force=True)
        if not content or "features" not in content:
            return jsonify({
                "status": "error",
                "message": "Payload JSON harus memiliki key 'features'"
            }), 400

        raw_features = content["features"]

        features_arr = np.array(raw_features, dtype=float)
        if features_arr.ndim == 1:
            features_arr = features_arr.reshape(1, -1)

        expected_len = len(feature_names) if feature_names else 42
        if features_arr.shape[1] < expected_len:
            padding = np.zeros((features_arr.shape[0], expected_len - features_arr.shape[1]))
            features_arr = np.hstack([features_arr, padding])

        predictions = model.predict(features_arr)
        pred_list = [float(np.round(p, 2)) for p in predictions]

        return jsonify({
            "status": "success",
            "model": model_name,
            "prediction": pred_list,
            "unit": "IDR"
        }), 200

    except Exception as e:
        logging.error("Gagal melakukan inferensi: %s", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
