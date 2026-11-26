# Proyek MLOps: Prediksi harga skin CS2

Repositori ini berisi implementasi sistem MLOps untuk memprediksi tren pergerakan harga aset digital (*skins*) Counter-Strike 2 di Steam Community Market berbasis *Continual Learning*, penanganan *Data Drift*, pengemasan kontainer Docker, otomatisasi pipeline CI/CD end-to-end dengan GitHub Actions, orkestrasi multi-layanan berbasis Docker Compose, serta *model serving* langsung dari MLflow Model Registry dengan *horizontal scaling* dan *load balancing* Nginx.

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
├── nginx/
│   └── nginx.conf
├── notebooks/
├── scripts/
│   ├── build_serving_image.sh
│   ├── stress_test.py
│   └── verify_orchestration.py
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
├── docker-compose.yaml
├── Dockerfile
├── Dockerfile.mlflow
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

---

## Multi-Container Service Orchestration (Docker Compose)

Pada tahap orkestrasi, sistem mengintegrasikan tiga layanan mandiri ke dalam satu ekosistem terpadu menggunakan **Docker Compose**: API inferensi model, server pelacakan terpusat MLflow, dan basis data metadata relasional PostgreSQL.

### 1. Arsitektur layanan terorkestrasi

| Layanan | Nama Kontainer | Port Host | Port Kontainer | Fungsi & Ketergantungan |
| :--- | :--- | :---: | :---: | :--- |
| `postgres` | `cs2-postgres-db` | 5432 | 5432 | Basis data relasional untuk penyimpanan metadata MLflow dan tabel riwayat log inferensi (`inference_logs`). Dilengkapi healthcheck periodik. |
| `mlflow-server` | `cs2-mlflow-server` | 5001 | 5000 | Server pelacakan eksperimen dan Model Registry terpusat dengan backend store PostgreSQL. Bergantung pada `postgres` (`service_healthy`). |
| `api-service` | `cs2-api-service` | 8080 | 5000 | REST API inferensi harga skin CS2. Bergantung pada `postgres` dan `mlflow-server` untuk pencatatan log dan validasi model. |

### 2. Mekanisme utama orkestrasi
* **Service Discovery**: Seluruh kontainer berada di dalam satu custom bridge network (`mlops-net`). Komunikasi antar-layanan menggunakan nama host DNS internal (`postgres:5432` dan `http://mlflow-server:5000`), mengeliminasi ketergantungan pada alamat IP acak.
* **Data Persistence**: Menggunakan Docker Named Volumes (`pg_data` untuk `/var/lib/postgresql/data` dan `mlflow_data` untuk `/mlruns`). Data metadata, log inferensi, dan artefak model tetap persisten saat kontainer dihentikan atau direstart.
* **Urutan Startup Deterministik**: Konfigurasi `depends_on` dengan evaluasi `condition: service_healthy` memastikan layanan basis data siap melayani koneksi sebelum MLflow server dan API inferensi dijalankan.

### 3. Panduan operasional Docker Compose
1. **Membangun dan menjalankan seluruh layanan (background mode):**
   ```bash
   docker compose up -d --build
   ```
2. **Memeriksa status dan kesehatan kontainer:**
   ```bash
   docker compose ps
   ```
3. **Menjalankan pengujian integrasi otomatis:**
   ```bash
   python scripts/verify_orchestration.py
   ```
4. **Pengujian persistensi data (simulasi kegagalan layanan):**
   - Hentikan kontainer database:
     ```bash
     docker compose stop postgres
     ```
   - Jalankan kembali kontainer database:
     ```bash
     docker compose start postgres
     ```
   - Verifikasi riwayat log prediksi tetap utuh:
     ```bash
     curl -s http://localhost:8080/logs
     ```
5. **Menghentikan seluruh layanan:**
   ```bash
   docker compose down
   ```

---

## Model Serving, Load Balancing & Horizontal Scaling (MLflow + Nginx)

Pada tahap *Serving & Scaling*, layanan inferensi tidak lagi menggunakan skrip API manual dengan file `model.pkl` lokal, melainkan dikemas menjadi Docker image langsung dari **MLflow Model Registry** (`models:/cs2-skin-price-predictor@production`) dan diskalakan menjadi **3 replika** di balik **Load Balancer Nginx** (algoritma *Round Robin*).

### 1. Arsitektur layanan serving ter-scale

| Layanan | Image / Kontainer | Port Host | Port Internal | Replika | Fungsi |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `postgres` | `postgres:15-alpine` (`cs2-postgres-db`) | 5432 | 5432 | 1 | Backend metadata store PostgreSQL untuk MLflow Tracking & Model Registry. |
| `mlflow-server` | `Dockerfile.mlflow` (`cs2-mlflow-server`) | 5001 | 5000 | 1 | MLflow Server & Model Registry terpusat dengan HTTP artifact proxy (`--serve-artifacts`). |
| `model-serving` | `cs2-mlflow-serving:v1` | - | 8080 | **3** | Kontainer inferensi MLflow (`pyfunc`) yang dibangun langsung dari Model Registry (`@production`). |
| `nginx` | `nginx:alpine` (`cs2-nginx-lb`) | 8080 | 80 | 1 | Reverse proxy & Load Balancer (*Round Robin*) yang mendistribusikan trafik ke 3 replika `model-serving`. |

### 2. Alur build Docker image dari MLflow Model Registry
1. **Menyalakan `postgres` dan `mlflow-server`:**
   ```bash
   docker compose up -d --build postgres mlflow-server
   ```
2. **Melatih model (dengan signature) dan mempromosikan ke `@production`:**
   ```bash
   MLFLOW_TRACKING_URI=http://localhost:5001 python3 src/train.py --model_type random_forest --n_estimators 250 --max_depth 16 --min_samples_split 2 --run_name rf_tuned_deep_serving
   python3 src/evaluate.py --threshold 0.85 --stage Production --tracking_uri http://localhost:5001
   ```
3. **Membangun image `cs2-mlflow-serving:v1` langsung dari MLflow Registry:**
   ```bash
   bash scripts/build_serving_image.sh
   ```

### 3. Menjalankan 3 replika & Load Balancer Nginx
1. **Menyalakan seluruh stack dengan 3 replika:**
   ```bash
   docker compose up -d --remove-orphans
   ```
2. **Memeriksa status 3 replika dan Nginx:**
   ```bash
   docker compose ps
   ```

### 4. Pengujian endpoint & distribusi beban
* **Cek kesehatan melalui Nginx (memeriksa header `X-Upstream-Addr`):**
  ```bash
  curl -i http://localhost:8080/ping
  ```
* **Kirim request inferensi format `dataframe_split` ke `/invocations`:**
  ```bash
  curl -i -X POST http://localhost:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{"dataframe_split": {"columns": ["day_of_week", "volume_clean", "price_delta", "volume_delta", "price_ma_3d", "price_ma_7d", ...], "data": [[...]]}}'
  ```
  **Contoh respon JSON:**
  ```json
  {"predictions": [65266.644, 2158241.928]}
  ```

### 5. Pengujian beban (*Stress Test*) & Ringkasan Benchmark
Menjalankan pengujian beban konkuren (150 request, 12 worker konkuren) terhadap endpoint `/invocations` melalui Nginx:
```bash
python3 scripts/stress_test.py --requests 150 --concurrency 12 --label "3-Replicas (Nginx Round Robin)"
```

| Metrik Pengujian | 1 Replika (Baseline) | 3 Replika (Nginx Round Robin) | Interpretasi |
| :--- | :---: | :---: | :--- |
| **Total Durasi** | 12.344 s | **9.368 s** | **24,1% lebih cepat** menyelesaikan 150 request konkuren |
| **Throughput (RPS)** | 12.15 req/s | **16.01 req/s** | **+31,8% peningkatan kapasitas** pemrosesan per detik |
| **Error Rate** | 0.00% (150/0) | **0.00% (150/0)** | Seluruh request valid dilayani tanpa kegagalan |
| **Latensi Avg** | 969.02 ms | **731.17 ms** | Rata-rata waktu respons turun **24,5%** |
| **Latensi p50 (Median)** | 960.89 ms | **650.62 ms** | Mayoritas request **32,3% lebih cepat** |
| **Latensi p95 / p99** | 1167.06 / 1204.11 ms | 1585.37 / 1729.16 ms | *Tail latency* lebih tinggi akibat kontensi CPU pada host Codespace 2 vCPU saat 3 model berjalan paralel |
| **Distribusi Beban** | 100.0% (1 replika) | **33.3% / 33.3% / 33.3%** (`50 / 50 / 50`) | Trafik terdistribusi merata dan menghilangkan *single point of failure* |