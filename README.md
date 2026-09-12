# Proyek MLOps: Prediksi Harga Skin CS2

Repositori ini berisi proyek MLOps untuk memprediksi pergerakan harga skin CS2 di Steam Community Market. 

Karena harga di market sering fluktuatif dan berubah drastis (*data drift*), sistem ini juga merancang strategi *continual learning* supaya model ML-nya bisa terus beradaptasi dengan tren data historis dan volume transaksi yang terbaru.

## Struktur Folder

Biar rapi, struktur direktori di repo ini mengikuti standar konvensi *Cookiecutter Data Science*:

* `configs/` : Tempat nyimpan file konfigurasi (seperti parameter model atau API key).
* `data/` : Buat nyimpan *raw data* sementara hasil tarikan dari Steam API.
* `docs/` : Catatan dan dokumentasi lanjutan proyek.
* `models/` : Folder untuk nyimpan model ML regresi yang sudah selesai dilatih.
* `notebooks/` : File Jupyter (*.ipynb*) untuk eksplorasi data (EDA) dan coret-coret eksperimen awal.
* `src/` : Kode sumber utama. Saat ini baru berisi file `main.go` yang bertugas mengeksekusi *data ingestion* harian dari Steam.
* `tests/` : Kumpulan *unit test* untuk memastikan pipeline berjalan aman.

## Cara menjalankan (via GitHub Codespaces)

Biar nggak ribet setup *environment* secara lokal, proyek ini sudah dikonfigurasi menggunakan **GitHub Codespaces**.

1. Buka repositori ini, klik tombol hijau **Code**.
2. Masuk ke tab **Codespaces**, lalu klik **Create codespace on main** (atau branch eksperimen yang sedang aktif).
3. Tunggu sebentar sampai proses *build* selesai. Dependensi seperti Python, Jupyter, dan Go sudah terinstal otomatis!
4. Untuk ngetes *script* penarikan datanya, buka terminal dan jalankan: 
   `go run src/main.go`