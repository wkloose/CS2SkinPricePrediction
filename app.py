import os
import pickle
import logging
import datetime
import numpy as np
import requests
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)

MODEL_PATH = "models/model.pkl" if os.path.exists("models/model.pkl") else "model.pkl"
logging.info("Memuat model dari: %s...", MODEL_PATH)

try:
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
    logging.info("Model '%s' berhasil dimuat. Total fitur yang diharapkan: %d", model_name, len(feature_names))
except Exception as e:
    logging.error("Gagal memuat model: %s", str(e))
    model = None
    feature_names = []
    model_name = "cs2-skin-price-predictor"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mlops_db")
DB_USER = os.getenv("DB_USER", "mlops_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlops_password")

def get_db_connection():
    """Membuat koneksi ke database PostgreSQL metadata."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=3
        )
        return conn
    except Exception as e:
        logging.warning("Koneksi DB belum tersedia (%s:%s): %s", DB_HOST, DB_PORT, str(e))
        return None

def init_db():
    """Inisialisasi tabel pencatatan inferensi jika belum ada."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS inference_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_name VARCHAR(100),
                        features_count INT,
                        predicted_price DOUBLE PRECISION,
                        unit VARCHAR(10)
                    );
                """)
                conn.commit()
                logging.info("Tabel inference_logs siap digunakan.")
        except Exception as e:
            logging.error("Gagal inisialisasi tabel: %s", str(e))
        finally:
            conn.close()

init_db()

def check_mlflow_status():
    """Memeriksa keterjangkauan MLflow tracking server via nama layanan."""
    try:
        resp = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=2)
        if resp.status_code == 200:
            return {"status": "connected", "uri": MLFLOW_TRACKING_URI}
    except Exception:
        pass
    try:
        resp = requests.get(f"{MLFLOW_TRACKING_URI}/", timeout=2)
        if resp.status_code == 200:
            return {"status": "connected", "uri": MLFLOW_TRACKING_URI}
    except Exception as e:
        return {"status": "disconnected", "uri": MLFLOW_TRACKING_URI, "error": str(e)}
    return {"status": "reachable", "uri": MLFLOW_TRACKING_URI}

def check_db_status():
    """Memeriksa status konektivitas PostgreSQL."""
    conn = get_db_connection()
    if conn:
        conn.close()
        return {"status": "connected", "host": DB_HOST, "database": DB_NAME}
    return {"status": "disconnected", "host": DB_HOST, "database": DB_NAME}

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    """Endpoint status kesehatan layanan inferensi dan service discovery."""
    mlflow_info = check_mlflow_status()
    db_info = check_db_status()
    is_healthy = model is not None and db_info["status"] == "connected"

    return jsonify({
        "status": "healthy" if is_healthy else "degraded",
        "service": "CS2 Skin Price Prediction API",
        "model": model_name,
        "features_count": len(feature_names),
        "service_discovery": {
            "mlflow": mlflow_info,
            "database": db_info
        }
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint inferensi harga skin CS2 dengan pencatatan log ke DB."""
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

        # Simpan log inferensi ke PostgreSQL
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    for p_val in pred_list:
                        cur.execute("""
                            INSERT INTO inference_logs (model_name, features_count, predicted_price, unit)
                            VALUES (%s, %s, %s, %s);
                        """, (model_name, expected_len, p_val, "IDR"))
                    conn.commit()
            except Exception as e:
                logging.warning("Gagal mencatat log inferensi: %s", str(e))
            finally:
                conn.close()

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

@app.route("/logs", methods=["GET"])
def get_logs():
    """Endpoint untuk audit dan pembuktian persistensi data PostgreSQL."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"status": "error", "message": "Database disconnected"}), 503
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, timestamp, model_name, features_count, predicted_price, unit
                FROM inference_logs ORDER BY id DESC LIMIT 10;
            """)
            rows = cur.fetchall()
            logs = [
                {
                    "id": r[0],
                    "timestamp": r[1].isoformat() if hasattr(r[1], 'isoformat') else str(r[1]),
                    "model": r[2],
                    "features_count": r[3],
                    "predicted_price": r[4],
                    "unit": r[5]
                }
                for r in rows
            ]
            return jsonify({
                "status": "success",
                "total_logs_returned": len(logs),
                "logs": logs
            }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)