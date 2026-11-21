#!/usr/bin/env python3
"""
verify_orchestration.py
Skrip pengujian integrasi orkestrasi multi-layanan Docker Compose (LK-09).
Menguji:
1. Kesehatan endpoint API inferensi (/health) dan service discovery (MLflow & PostgreSQL)
2. Inferensi prediksi harga skin CS2 (/predict) dan pencatatan otomatis ke DB
3. Validasi persistensi data log inferensi (/logs)
"""

import sys
import time
import requests
import json

API_BASE = "http://localhost:8080"
MLFLOW_BASE = "http://localhost:5001"

def test_health():
    print("\n[1/4] Menguji Endpoint /health dan Service Discovery...")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"Status Code: {resp.status_code}")
        data = resp.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2))
        assert resp.status_code == 200, "Health endpoint gagal!"
        assert data.get("service_discovery", {}).get("database", {}).get("status") == "connected", "Koneksi DB gagal!"
        print("--> Uji Health & Service Discovery LULUS (Semua terhubung via nama layanan)!\n")
        return True
    except Exception as e:
        print(f"--> ERROR: {e}")
        return False

def test_predict():
    print("[2/4] Menguji Endpoint /predict dan Pencatatan Log ke DB...")
    payload = {
        "features": [
            210000.0, 1250.0, 205000.0, 215000.0, 212000.0, 208000.0,
            0.024, 0.05, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    }
    try:
        resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
        print(f"Status Code: {resp.status_code}")
        data = resp.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2))
        assert resp.status_code == 200, "Predict endpoint gagal!"
        assert "prediction" in data, "Key prediction tidak ditemukan!"
        print("--> Uji Prediksi LULUS!\n")
        return True
    except Exception as e:
        print(f"--> ERROR: {e}")
        return False

def test_logs():
    print("[3/4] Menguji Endpoint /logs untuk Memverifikasi Data Tersimpan di DB...")
    try:
        resp = requests.get(f"{API_BASE}/logs", timeout=5)
        print(f"Status Code: {resp.status_code}")
        data = resp.json()
        print("Response JSON:")
        print(json.dumps(data, indent=2))
        assert resp.status_code == 200, "Logs endpoint gagal!"
        assert data.get("total_logs_returned", 0) > 0, "Log tidak tercatat di DB!"
        print("--> Uji Logging Data LULUS!\n")
        return True
    except Exception as e:
        print(f"--> ERROR: {e}")
        return False

def test_mlflow_ui():
    print("[4/4] Menguji Akses Langsung ke MLflow Tracking Server UI (Port 5001)...")
    try:
        resp = requests.get(f"{MLFLOW_BASE}/", timeout=5)
        print(f"Status Code: {resp.status_code}")
        assert resp.status_code == 200, "MLflow UI tidak merespon 200!"
        print("--> MLflow Tracking Server UI Berjalan Normal!\n")
        return True
    except Exception as e:
        print(f"--> ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  VERIFIKASI INTEGRASI ORKESTRASI DOCKER COMPOSE (LK-09)")
    print("=" * 60)
    success = test_health() and test_predict() and test_logs() and test_mlflow_ui()
    if success:
        print("SEMUA PENGUJIAN INTEGRASI BERHASIL 100%!")
        sys.exit(0)
    else:
        print("BEBERAPA PENGUJIAN GAGAL.")
        sys.exit(1)
