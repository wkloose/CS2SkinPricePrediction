# Proyek MLOps: Prediksi harga skin CS2

Repositori ini berisi implementasi sistem MLOps untuk memprediksi tren pergerakan harga aset digital (*skins*) Counter-Strike 2 di Steam Community Market berbasis *Continual Learning*, penanganan *Data Drift*, pengemasan kontainer Docker, dan otomatisasi pipeline CI/CD end-to-end dengan GitHub Actions.

---

## Struktur direktori

Struktur proyek mengadopsi standar industri *Cookiecutter Data Science*:

```text
CS2SkinPricePrediction/
├── .devcontainer/
├── .github/
│   └── workflows/
│       └── mlops-automation.yaml
├── configs/
│   └── items.json
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
│   └── model.pkl
├── notebooks/
├── src/
│   ├── evaluate.py
│   ├── export_model.py
│   ├── ingest_data.py
│   ├── prepare_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── main.go
├── tests/
│   └── test_preprocess.py
├── app.py
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Data Version Control (DVC) Pipeline

Proyek ini menggunakan DVC (Data Version Control) untuk mengelola silsilah data (*data lineage*) dan melacak versi dataset mentah secara efisien tanpa membebani repositori Git dengan file data berukuran besar.

1. **Struktur pelacakan data**
   - File data mentah di folder `data/raw/` diabaikan oleh Git via `.gitignore` dan dikelola sepenuhnya oleh DVC.
   - File pointer metadata `data/raw.dvc` melacak hash MD5 dan volume dataset di repositori Git.
   - Seluruh versi fisik dataset disimpan di dalam remote storage DVC.

2. **Panduan reproduksibilitas data**
   - Mengunduh dataset yang sesuai dengan commit: `dvc pull`

---

## Machine Learning experimentation & tracking (MLflow)

Proyek ini mengintegrasikan **MLflow Tracking** untuk mengelola siklus hidup eksperimentasi model regresi harga skin Counter-Strike 2, memastikan seluruh parameter input, metrik evaluasi, dan artefak model tersimpan secara terstruktur dan dapat direproduksi (*reproducible*).

### 1. Tiga pilar pelacakan eksperimen
* **Parameters**: Dicatat melalui `mlflow.log_params()` mencakup jenis algoritma (`model_type`), parameter regularisasi (`alpha`), serta struktur pohon keputusan (`n_estimators`, `max_depth`, `min_samples_split`).
* **Metrics**: Dicatat melalui `mlflow.log_metrics()` dengan metrik regresi komprehensif:
  * **RMSE** (*Root Mean Squared Error*)
  * **MAE** (*Mean Absolute Error*)
  * **R² Score** (*Coefficient of Determination*)
  * **MAPE** (*Mean Absolute Percentage Error*)
* **Artifacts**: Disimpan secara otomatis meliputi plot evaluasi *actual vs predicted price* (`artifacts/evaluation_plots/`) dan serialisasi model biner Scikit-Learn (`model/MLmodel`, `model.pkl`).

### 2. Ringkasan hasil eksperimen

| Nama Run | Algoritma | Parameter Kunci | RMSE (IDR) | MAE (IDR) | R² Score | MAPE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `ridge_baseline_20261018` | Ridge Regression | `alpha=1.0` | ~168.420 | ~112.350 | ~0.8412 | ~0.2450 |
| `rf_baseline_20261018` | Random Forest | `n_est=100, depth=8` | ~108.430 | ~64.650 | 0.9386 | 0.1290 |
| `rf_tuned_deep_20261018` | Random Forest | `n_est=250, depth=16` | **~79.007** | **~48.641** | **0.9674** | **0.1058** |

> **Justifikasi model terbaik**: Model **`rf_tuned_deep_20261018`** terpilih sebagai kandidat terbaik karena menghasilkan skor **R² tertinggi (0.9674)** dengan nilai galat terendah (MAE Rp 48.641 dan MAPE 10,58%).

### 3. Panduan reproduksi eksperimen
Menjalankan pelatihan model:
```bash
python src/train.py --model_type random_forest --n_estimators 250 --max_depth 16 --min_samples_split 2 --run_name rf_tuned_deep_20261018
```

---

## Model packaging & containerization (Docker)

Proyek ini mengemas model terbaik hasil eksperimentasi MLflow (`rf_tuned_deep_20261018`) ke dalam layanan inferensi REST API mandiri menggunakan **Flask** dan **Docker**, menjamin reproduksibilitas penuh di lingkungan apa pun tanpa risiko *dependency hell*.

### 1. Struktur layanan API
* **`GET /health`**: Memeriksa status kesehatan layanan inferensi dan model yang aktif.
* **`POST /predict`**: Menerima payload JSON fitur skin CS2 dan mengembalikan estimasi harga pasar (IDR).

### 2. Panduan menjalankan kontainer
1. **Membangun docker image:**
   ```bash
   docker build -t cs2-price-api:v1 .
   ```
2. **Menjalankan kontainer (background mode):**
   ```bash
   docker run -d -p 5000:5000 --name cs2-service cs2-price-api:v1
   ```
3. **Memeriksa status & log:**
   ```bash
   docker ps
   docker logs cs2-service
   ```

### 3. Pengujian endpoint inferensi
Kirim permintaan inferensi menggunakan `curl`:
```bash
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 250, 1500.0, 50, 210000.0, 215000.0]}'
```

**Respon JSON:**
```json
{
  "model": "rf_tuned_deep_20261018",
  "prediction": [208768.77],
  "status": "success",
  "unit": "IDR"
}
```

### 4. Menghentikan layanan
```bash
docker stop cs2-service
docker rm cs2-service
```

---

## CI/CD Pipeline Automation (GitHub Actions)

Otomatisasi pengujian, pelatihan, dan registrasi model secara berkelanjutan dikelola melalui alur kerja GitHub Actions pada berkas `.github/workflows/mlops-automation.yaml` dengan prinsip Code as a Trigger.

### 1. Tiga tahap pipeline CI/CD
* **Job 1: Automated Unit Testing (`testing`)**: Menjalankan pengujian otomatis dengan pytest pada `tests/test_preprocess.py` untuk memvalidasi sanitasi harga, volume, serta integritas fitur pasar sebelum tahap pelatihan.
* **Job 2: Model Training & Experiment Tracking (`training`)**: Berjalan setelah testing lulus (`needs: testing`). Menyiapkan dataset fitur terverifikasi, melatih model Random Forest, mencatat metrik ke MLflow, dan mengunggah artefak model.
* **Job 3: Threshold Evaluation & Model Registry (`registry`)**: Berjalan setelah training selesai (`needs: training`). Mengevaluasi metrik model terbaru terhadap ambang batas minimal (R² >= 0.85). Model yang memenuhi syarat didaftarkan ke MLflow Model Registry dan statusnya dipromosikan ke tahap Staging.

### 2. Panduan eksekusi lokal
1. Menjalankan pengujian unit:
```bash
pytest tests/ -v
```
2. Menyiapkan dataset fitur:
```bash
python src/prepare_data.py
```
3. Melatih model dengan MLflow:
```bash
python src/train.py --model_type random_forest --n_estimators 150 --max_depth 12 --run_name manual_test_run
```
4. Mengevaluasi threshold dan registrasi model:
```bash
python src/evaluate.py --threshold 0.85
```