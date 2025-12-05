
# Penelitian Fetal Heart Rate Generative Adversarial Network (GAN)

Repository ini berisi seluruh kode dan skrip penelitian terkait generasi dan analisis sinyal Fetal Heart Rate (FHR) menggunakan metode Generative Adversarial Network (GAN). Penelitian ini dikembangkan untuk keperluan skripsi dan eksperimen augmentasi data FHR.

## 🧠 Latar Belakang

Pemantauan FHR penting dalam mendeteksi kondisi janin seperti hipoksia dan distress fetal. Namun, data FHR yang tersedia sering kali terbatas serta tidak seimbang antara kelas normal dan patologis. Hal ini menyulitkan pelatihan model pembelajaran mesin.

Generative Adversarial Network (GAN) digunakan untuk menghasilkan sinyal FHR sintetik yang realistis, sehingga dapat memperkaya dataset dan meningkatkan performa model klasifikasi di penelitian selanjutnya.

## 📁 Struktur Repository

File dan folder utama dalam repository:

- `LoadDataset.py`, `LoadDatasetph72.py` — memuat dan memproses dataset FHR.
- `preprocesscode.py` — preprocessing sinyal (normalisasi, pembersihan noise, segmentasi).
- `windowed_dataset.py` — pembuatan window sinyal FHR untuk pelatihan.
- `CNNcodeCTGGAN.py`, `cnncodefhrgan.py`, `cnnmodels_fixed.py` — model CNN dan arsitektur pendukung.
- `NewFHRGANmodel131018nov.py` — implementasi model FHRGAN.
- `CTGGANTRAIN.py`, `CTGGANTRAINPH72.py` — skrip pelatihan CTGGAN.
- `NEWFHRGANTRAIN.py`, `NEWFHRGANTRAINph72.py` — skrip pelatihan FHRGAN.
- `EVALCODE.py` — evaluasi kualitas sinyal sintetis.
- `sinyal2.py`, `view_signal_notebook.py`, `view_signal_run_code.py` — visualisasi sinyal asli dan sintetik.

## ⚙️ Instalasi

1. Python 3.8 atau lebih baru.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Siapkan dataset FHR dan sesuaikan path pada file `LoadDataset.py`.

## 🚀 Cara Menggunakan

### 1. Preprocess Data  
```bash
python preprocesscode.py
```

### 2. Membuat Window Dataset  
```bash
python windowed_dataset.py
```

### 3. Melatih Model GAN  
```bash
python CTGGANTRAIN.py
```
atau  
```bash
python NEWFHRGANTRAIN.py
```

### 4. Menghasilkan Sinyal Sintetik  
Gunakan model generator yang telah terlatih.

### 5. Evaluasi  
```bash
python EVALCODE.py
```

### 6. Visualisasi  
```bash
python view_signal_run_code.py
```

## 📊 Tujuan & Aplikasi

- Augmentasi sinyal fisiologis menggunakan GAN.
- Mengatasi ketidakseimbangan kelas.
- Mendukung penelitian medis dan time-series generatif.

## 📚 Referensi

- Puspitasari et al. — GAN for unbalanced FHR classification.
- Yu et al. — CTGGAN architecture.

## 📌 Catatan

Pastikan kualitas data input baik, dan selalu lakukan evaluasi kualitatif & kuantitatif.

## 📎 Lisensi

Repository ini dibuat untuk tujuan akademik.
