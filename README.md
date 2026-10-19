# Proyek MLOps: Prediksi Harga Skin CS2

Repositori ini berisi implementasi sistem MLOps untuk memprediksi tren pergerakan harga aset digital (*skins*) Counter-Strike 2 di steam community market berbasis *Continual Learning* dan penanganan *Data Drift*.

---

## Struktur direktori

Struktur proyek mengadopsi standar industri *Cookiecutter data science*:

```text
CS2SkinPricePrediction/
├── .devcontainer/       # Konfigurasi container gitHub codespaces
├── configs/
│   └── items.json       # Konfigurasi modular daftar skin CS2
├── data/
│   ├── raw/             # Snapshot data mentah bertimestamp dari steam API
│   └── processed/       # Dataset hasil feature engineering (CSV & parquet)
├── docs/                # Dokumentasi arsitektur dan lembar kerja MLOps
├── models/              # Artefak model ML (tahap berikutnya)
├── notebooks/           # Notebook eksplorasi data awal (EDA)
├── src/
│   ├── ingest_data.py   # Pipeline penarikan data dinamis & rate-limiting
│   ├── preprocess.py    # Pipeline pembersihan data & feature engineering
│   └── main.go          # Pipeline ingestion alternatif berbasis Go
├── requirements.txt     # Daftar dependensi pustaka Python
└── README.md            # Dokumentasi utama proyek


Data Version Control (DVC) Pipeline

Proyek ini menggunakan DVC (Data Version Control) untuk mengelola silsilah data (data lineage) dan melacak versi dataset mentah secara efisien tanpa membebani repositori Git dengan file data berukuran besar.

1. Struktur Pelacakan Data
- File data mentah di folder `data/raw/` diabaikan oleh Git via `.gitignore` dan dikelola sepenuhnya oleh DVC.
- File pointer metadata `data/raw.dvc` melacak hash MD5 dan volume dataset di repositori Git.
- Seluruh versi fisik dataset disimpan di dalam remote storage DVC.

2. Panduan Reproduksibilitas Data
Mengunduh dataset yang sesuai dengan commit: dvc pull


##  Machine learning experimentation & tracking (MLflow)

Proyek ini mengintegrasikan **MLflow tracking** untuk mengelola siklus hidup eksperimentasi model regresi harga skin Counter-Strike 2, memastikan seluruh parameter input, metrik evaluasi, dan artefak model tersimpan secara terstruktur dan dapat direproduksi (*reproducible*).

### 1. Tiga pilar pelacakan eksperimen
* **Parameters**: Dicatat melalui `mlflow.log_params()` mencakup jenis algoritma (`model_type`), parameter regularisasi (`alpha`), serta struktur pohon keputusan (`n_estimators`, `max_depth`, `min_samples_split`).
* **Metrics**: Dicatat melalui `mlflow.log_metrics()` dengan metrik regresi komprehensif:
  * **RMSE** (*Root Mean Squared Error*)
  * **MAE** (*Mean Absolute Error*)
  * **R² Score** (*Coefficient of Determination*)
  * **MAPE** (*Mean Absolute Percentage Error*)
* **Artifacts**: Disimpan secara otomatis meliputi plot evaluasi *actual vs predicted price* (`artifacts/evaluation_plots/`) dan serialisasi model biner Scikit-Learn (`model/MLmodel`, `model.pkl`).

### 2. Ringkasan Hasil Eksperimen

| Nama Run | Algoritma | Parameter Kunci | RMSE (IDR) | MAE (IDR) | R² Score | MAPE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `ridge_baseline_20261018` | Ridge Regression | `alpha=1.0` | ~168.420 | ~112.350 | ~0.8412 | ~0.2450 |
| `rf_baseline_20261018` | Random Forest | `n_est=100, depth=8` | ~108.430 | ~64.650 | 0.9386 | 0.1290 |
| `rf_tuned_deep_20261018` | Random Forest | `n_est=250, depth=16` | **~79.007** | **~48.641** | **0.9674** | **0.1058** |

> **Justifikasi model terbaik**: Model **`rf_tuned_deep_20261018`** terpilih sebagai kandidat terbaik karena menghasilkan skor **R² tertinggi (0.9674)** dengan nilai galat terendah (MAE Rp 48.641 dan MAPE 10,58%). Model ini siap digunakan untuk tahap pengemasan (*Model Packaging*) pada milestone berikutnya.

### 3. Panduan reproduksi eksperimen & MLflow UI
* **Menjalankan pelatihan model:**
  ```bash
  python src/train.py --model_type random_forest --n_estimators 250 --max_depth 16 --min_samples_split 2 --run_name rf_tuned_deep_20261018
