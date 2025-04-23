# Melanoma Prediction Web App

Aplikasi web untuk prediksi melanoma menggunakan model hybrid EfficientNet-B5 + ViT dengan metadata.

## Struktur File

```
melanomaPredict/
├── app.py                 # Aplikasi Flask
├── requirements.txt       # Dependencies
├── templates/            
│   └── index.html        # Template halaman web
└── model/                # Folder untuk menyimpan model weights
    └── model.pth  # Model weights
```

## Setup

1. **Persiapkan Environment**

   ```bash
   # Buat virtual environment
   python -m venv venv
   
   # Aktifkan virtual environment
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Konfigurasi Path**

   Buka `app.py` dan sesuaikan path model:

   ```python
   MODEL_PATH = 'path/to/your/effnetb5_384_9c_50epo_ext_BEST_epoch48.pth'
   ```

   Pastikan file model weights tersedia di path yang ditentukan.

## Menjalankan Aplikasi

1. **Jalankan Server Flask**

   ```bash
   python app.py
   ```

2. **Akses Aplikasi**

   Buka browser dan akses:
   ```
   http://localhost:5000
   ```

## Penggunaan

1. Upload gambar lesi kulit
2. Isi informasi metadata:
   - Umur pasien (0-120)
   - Jenis kelamin (Pria/Wanita)
   - Lokasi lesi
3. Klik "Analyze Image"
4. Hasil akan menampilkan:
   - Top 3 prediksi dengan probabilitas
   - Visualisasi Grad-CAM
   - Peringatan jika terdeteksi risiko melanoma tinggi

## Catatan Penting

- Aplikasi ini menggunakan GPU jika tersedia, namun akan fallback ke CPU jika tidak ada GPU
- Model membutuhkan metadata lengkap untuk hasil optimal
- Visualisasi Grad-CAM membantu interpretasi area yang menjadi fokus model
- Aplikasi ini untuk tujuan edukasi, bukan untuk diagnosis medis

## Troubleshooting

1. **Import Error saat Install Requirements**
   ```bash
   # Jika ada masalah dengan torch, install manual:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # Untuk CUDA 11.8
   ```

2. **CUDA Out of Memory**
   ```python
   # Di app.py, kurangi batch_size jika terjadi OOM:
   BATCH_SIZE = 1  # Default untuk inference
   ```

3. **Model Loading Error**
   - Pastikan path model benar
   - Pastikan format model compatible (PyTorch state dict)
   - Cek GPU memory jika menggunakan CUDA

## Requirements

Lihat `requirements.txt` untuk daftar lengkap dependencies. Key packages:
- Flask
- PyTorch
- Transformers
- Timm
- Albumentations
- PyTorch-Grad-CAM 
