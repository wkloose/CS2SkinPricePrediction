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

